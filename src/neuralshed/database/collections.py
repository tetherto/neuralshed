"""Read/write API for dataset collections.

A *collection* is a saved group of datasets assembled to train one decoder together, plus the
``CombineStudies`` step chain that pools them. The persisted form of a decoder's spec. Its
``target_event_types`` say what the decoder outputs, and so which head this is. Per-study
filters belong on the study step of the collection chain, not here.

Writes live here so ``query.py`` stays a read-only view over the catalog. Assembled
interactively in the collection builder app, or programmatically::

    from neuralshed.database import collections
    collections.save_collection(
        collection_name="speech-text",
        members=["card_2024", "willet_2023", "fan_2023"],
        input_event_types=("SpikeCounts", "BandPower"),
        target_event_types=("Sentence",),
    )
    collections.list_collections()
    collections.get_collection(collection_name="speech-text")   # → dict (fields + members + chain)
    collections.load_chain(collection_name="speech-text")       # → runnable Chain, paths resolved
    collections.delete_collection(collection_name="speech-text")

Saving is cheap and touches no data. Turning a collection into something trainable needs its
*manifest*, the resolved table of what each recording actually contains, and that is a separate
call because it runs the chain and opens data files::

    collections.manifest_status(collection_name="speech-text")  # not_built | stale | built
    collections.build_manifest(collection_name="speech-text")   # slow, idempotent, stores the blob
    collections.get_manifest(collection_name="speech-text")     # fast, offline
"""

import copy
import typing as tp
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from . import db
from .schema import Collection, CollectionDataset, Dataset

if tp.TYPE_CHECKING:
    import neuralset as ns

    from neuralshed.manifest import CollectionManifest


def save_collection(
    collection_name: str,
    members: tp.Iterable[str],
    *,
    description: str | None = None,
    notes: str | None = None,
    chain: dict | None = None,
    input_event_types: tp.Sequence[str] | None = None,
    target_event_types: tp.Sequence[str] | None = None,
    check_streams: bool = True,
    replace: bool = True,
) -> int:
    """Create (or replace, if it exists) a collection; returns its id.

    ``members`` is a list of dataset ids. ``chain`` is the composed neuroai step chain (see
    ``Collection.chain``). Pass ``replace=False`` to raise instead of overwriting an existing
    name.

    ``input_event_types`` and ``target_event_types`` are neuroai event class names, the declared
    intention: what this collection reads and what it decodes. They are checked against the
    catalog before saving (see :func:`check_declared_streams`), which needs no data drive, so a
    member that cannot contribute is caught here rather than as a mysteriously empty extraction
    later. Pass ``check_streams=False`` to save a collection the catalog does not yet describe.

    Saving does **not** build a manifest. Curating a collection and materializing one are
    different activities with different latencies, and folding the slow one into the fast one
    makes it invisible. Call :func:`build_manifest` when you want it.
    """
    members = list(members)
    if check_streams and (input_event_types or target_event_types):
        report = check_declared_streams(
            dataset_ids=members,
            input_event_types=input_event_types,
            target_event_types=target_event_types,
        )
        if report["missing_targets"]:
            raise ValueError(
                "members cannot supply the declared targets "
                f"{sorted(target_event_types or ())}: {report['missing_targets']}; "
                "targets are required"
            )
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        existing = s.scalar(
            select(Collection).where(Collection.name == collection_name)
        )
        if existing is not None:
            if not replace:
                raise ValueError(
                    f"collection {collection_name!r} already exists (replace=False)"
                )
            s.delete(existing)
            s.flush()
        col = Collection(
            name=collection_name,
            description=description,
            notes=notes,
            chain=chain,
            input_event_types=list(input_event_types) if input_event_types else None,
            target_event_types=list(target_event_types) if target_event_types else None,
        )
        for dataset_id in members:
            col.members.append(CollectionDataset(dataset_id=dataset_id))
        s.add(col)
        s.commit()
        return col.id


def list_collections() -> pd.DataFrame:
    """One row per collection with its member datasets rolled up."""
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        cols = s.scalars(select(Collection)).all()
        rows = [
            {
                "id": c.id,
                "name": c.name,
                "decodes": ", ".join(c.target_event_types or ()),
                "manifest": "built" if c.manifest else "not_built",
                "n_datasets": len(c.members),
                "datasets": ", ".join(sorted(m.dataset_id for m in c.members)),
            }
            for c in cols
        ]
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "name",
            "decodes",
            "manifest",
            "n_datasets",
            "datasets",
        ],
    )


def get_collection(collection_name: str) -> dict | None:
    """Full collection as a dict (fields + ordered ``datasets`` ids), or ``None``.

    ``chain`` is the composed step chain, carrying no ``path``. Use :func:`load_chain` to
    resolve paths and get a runnable ``Chain``.
    """
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        c = s.scalar(select(Collection).where(Collection.name == collection_name))
        if c is None:
            return None
        return {
            "name": c.name,
            "description": c.description,
            "notes": c.notes,
            "chain": c.chain,
            "input_event_types": list(c.input_event_types or ()),
            "target_event_types": list(c.target_event_types or ()),
            "datasets": [m.dataset_id for m in c.members],
        }


def delete_collection(collection_name: str) -> bool:
    """Delete a collection (and its membership rows); ``True`` if one existed."""
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        c = s.scalar(select(Collection).where(Collection.name == collection_name))
        if c is None:
            return False
        s.delete(c)
        s.commit()
        return True


def check_declared_streams(
    dataset_ids: tp.Iterable[str],
    input_event_types: tp.Sequence[str] | None = None,
    target_event_types: tp.Sequence[str] | None = None,
) -> dict:
    """Which members can supply the declared event classes, according to the catalog.
    Two different rules:
    - inputs are wanted-if-present
    - targets are required
    """
    from .schema import EventType

    ids = list(dataset_ids)
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        known = set(s.scalars(select(Dataset.id).where(Dataset.id.in_(ids))))
        unknown = sorted(set(ids) - known)
        if unknown:
            raise KeyError(
                f"unknown dataset(s) {unknown}: not in the `datasets` catalog"
            )
        rows = s.execute(
            select(EventType.dataset_id, EventType.event_class).where(
                EventType.dataset_id.in_(ids)
            )
        ).all()
    available: dict[str, set[str]] = {i: set() for i in ids}
    for dataset_id, event_class in rows:
        if event_class:
            available[dataset_id].add(event_class)

    wanted_in = set(input_event_types or ())
    wanted_out = set(target_event_types or ())
    return {
        "available": {i: sorted(v) for i, v in available.items()},
        "inputs": {i: sorted(wanted_in & v) for i, v in available.items()},
        # a member supplying none of the declared inputs cannot be read at all
        "missing_inputs": sorted(
            i for i, v in available.items() if wanted_in and not (wanted_in & v)
        ),
        "missing_targets": sorted(
            i for i, v in available.items() if wanted_out and not (wanted_out & v)
        ),
    }


def manifest_status(collection_name: str) -> str:
    """``not_built``, ``stale`` or ``built``.

    Derived rather than stored. A status column would be a second thing that can be wrong: it
    could read "built" beside a blob that no longer matches its recipe. Comparing two digests
    costs nothing and cannot disagree with itself.

    Staleness here means the *recipe* changed. Sessions appearing on the drive after a build are
    deliberately not noticed, so picking them up needs ``build_manifest(force=True)``.
    """
    col = get_collection(collection_name=collection_name)
    if col is None:
        raise KeyError(f"no collection named {collection_name!r}")
    stored = _stored_manifest_meta(collection_name=collection_name)
    if not stored["manifest"]:
        return "not_built"
    from neuralshed.manifest import BUILDER_VERSION

    if stored["builder_version"] != BUILDER_VERSION:
        return "stale"
    return "built" if stored["recipe_digest"] == _recipe_digest(col) else "stale"


def get_manifest(collection_name: str) -> "CollectionManifest":
    """The stored manifest, or a clear error naming the fix."""
    from neuralshed.manifest import CollectionManifest

    status = manifest_status(collection_name=collection_name)
    if status != "built":
        raise ValueError(
            f"collection {collection_name!r} has a {status} manifest; run "
            f"collections.build_manifest(collection_name={collection_name!r}) first"
        )
    stored = _stored_manifest_meta(collection_name=collection_name)
    return CollectionManifest.from_dict(stored["manifest"])


def build_manifest(
    collection_name: str,
    *,
    force: bool = False,
    must_exist: bool = True,
    **resolve: tp.Any,
) -> "CollectionManifest":
    """Resolve the collection's layout table and store it, returning it either way.

    This is the slow step, and it is separate from :func:`save_collection` on purpose: it runs
    the whole chain and opens data files to count channels. Idempotent, so it is safe to call
    from anywhere; when the recipe has not changed it returns the stored blob untouched.

    ``force=True`` rescans regardless, which is how sessions added since the last build are
    picked up.

    ``**resolve`` is passed through to :meth:`CollectionManifest.from_events`, which is how the
    per-member facts a collection does not carry itself (``participants``, ``regions``) reach the
    resolved table.
    """
    from neuralshed.manifest import CollectionManifest

    col = get_collection(collection_name=collection_name)
    if col is None:
        raise KeyError(f"no collection named {collection_name!r}")
    if not force and manifest_status(collection_name=collection_name) == "built":
        return get_manifest(collection_name=collection_name)
    if not col["input_event_types"]:
        raise ValueError(
            f"collection {collection_name!r} declares no input_event_types, so there is "
            "nothing to resolve; save it with input_event_types=(...) first"
        )

    chain = load_chain(collection_name=collection_name, must_exist=must_exist)
    events = chain.run().reset_index(drop=True)
    manifest = CollectionManifest.from_events(
        events,
        inputs=col["input_event_types"],
        targets=col["target_event_types"],
        recipe_digest=_recipe_digest(col),
        built_at=_utc_now(),
        **resolve,
    )
    _store_manifest(collection_name=collection_name, manifest=manifest)
    return manifest


def _recipe_digest(col: dict, **extra: tp.Any) -> str:
    """Everything that changes what the manifest would resolve to.

    The keyword names here are hashed, so they are a wire format rather than an API: renaming
    one moves every stored digest and marks every built manifest stale.
    """
    from neuralshed.manifest import recipe_digest

    return recipe_digest(
        chain=col["chain"],
        inputs=sorted(col["input_event_types"]),
        targets=sorted(col["target_event_types"]),
        datasets=sorted(col["datasets"]),
        **extra,
    )


def _utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _stored_manifest_meta(collection_name: str) -> dict:
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        c = s.scalar(select(Collection).where(Collection.name == collection_name))
        if c is None:
            raise KeyError(f"no collection named {collection_name!r}")
        return {
            "manifest": c.manifest,
            "built_at": c.manifest_built_at,
            "recipe_digest": c.manifest_recipe_digest,
            "builder_version": c.manifest_builder_version,
        }


def _store_manifest(collection_name: str, manifest: "CollectionManifest") -> None:
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        c = s.scalar(select(Collection).where(Collection.name == collection_name))
        if c is None:
            raise KeyError(f"no collection named {collection_name!r}")
        c.manifest = manifest.to_dict()
        c.manifest_built_at = manifest.built_at
        c.manifest_recipe_digest = manifest.recipe_digest
        c.manifest_builder_version = manifest.builder_version
        s.commit()


def inject_paths(chain: dict, paths: tp.Mapping[str, str | Path]) -> dict:
    """A copy of a collection config with each member's ``Study`` given its ``path``.

    ``paths`` maps a study class name to that dataset's data directory. Pure: it touches neither
    the database nor the filesystem, so it can validate a config on a machine with no data.
    """
    cfg = copy.deepcopy(chain)
    try:
        chains = cfg["steps"]["CombineStudies"]["chains"]
    except (KeyError, TypeError) as e:
        raise ValueError("config is not a Chain headed by CombineStudies") from e
    for key, member in chains.items():
        head = member[0] if isinstance(member, list) else member
        cls = head.get("name") if isinstance(head, dict) else None
        if cls not in paths:
            raise KeyError(f"no path for member {key!r} (study class {cls!r})")
        head["path"] = str(paths[cls])
    return cfg


def data_dirs(
    dataset_ids: tp.Iterable[str],
    *,
    must_exist: bool = False,
) -> dict[str, Path]:
    """Map each dataset's study class to ``<NEURALSHED_DATA_ROOT>/<dataset_id>/data``.

    A config names its members by study class, so that is what the mapping is keyed on. The
    ``Study`` base appends the class-name subfolder itself, hence ``data/`` and not the leaf.
    """
    from ..datasets.utils import dataset_dir

    ids = list(dataset_ids)
    Session = sessionmaker(bind=db.get_engine(), future=True)
    with Session() as s:
        rows = s.execute(
            select(Dataset.id, Dataset.study_class).where(Dataset.id.in_(ids))
        ).all()
    found = {i for i, _ in rows}
    if missing := set(ids) - found:
        raise KeyError(f"unknown dataset(s): {sorted(missing)}")
    out = {}
    for dataset_id, study_class in rows:
        if study_class is None:
            raise ValueError(f"dataset {dataset_id!r} has no study_class")
        out[study_class] = dataset_dir(dataset_id, must_exist=must_exist) / "data"
    return out


def load_chain(collection_name: str, *, must_exist: bool = False) -> "ns.Step":
    """A saved collection's chain, runnable, with paths resolved against this machine.

    Pass ``must_exist=True`` to fail up front with a clear error when the data drive is not
    mounted. Either way, constructing a ``Study`` touches the filesystem and creates a missing
    ``<root>/<id>/data``; :func:`inject_paths` is the filesystem-free route to a config dict.
    """
    import neuralset as ns

    from neuralshed import (
        collection,
        studies,
        transforms,
    )  # noqa: F401 — registers classes

    col = get_collection(collection_name=collection_name)
    if col is None:
        raise KeyError(f"no collection named {collection_name!r}")
    if col["chain"] is None:
        raise ValueError(f"collection {collection_name!r} has no composed step chain")
    return ns.Step(
        **inject_paths(
            chain=col["chain"],
            paths=data_dirs(dataset_ids=col["datasets"], must_exist=must_exist),
        )
    )


__all__ = [
    "save_collection",
    "check_declared_streams",
    "manifest_status",
    "build_manifest",
    "get_manifest",
    "list_collections",
    "get_collection",
    "delete_collection",
    "inject_paths",
    "data_dirs",
    "load_chain",
]
