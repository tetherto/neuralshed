"""Populate the metadata database from per-dataset declarations.

Each ``datasets/<name>/metadata.py`` exposes a ``get_metadata() -> dict`` accessor that
returns the dataset's full payload: the static catalog (``DATASET_METADATA``) with the
per-session rows folded in under ``"sessions"`` (empty when the data drive isn't mounted).

This is the interim stand-in for the eventual neuroai ``Study`` objects; the dict shape
mirrors what a ``Study`` will emit, so a ``from_study()`` path can replace the
declarations without a schema change.

CLI::

    python -m neuralshed.database.populate                 # catalog + sessions (if drive mounted)
    python -m neuralshed.database.populate --catalog-only  # skip sessions
    python -m neuralshed.database.populate --datasets card_2024,willet_2021
    python -m neuralshed.database.populate --db sqlite:////tmp/test.db
"""

import argparse
import importlib
import sys
from pathlib import Path

from sqlalchemy.orm import Session as OrmSession

from ..datasets import participants

from . import db, vocab
from .schema import (
    Array,
    BrainArea,
    Dataset,
    EventType,
    Participant,
    Subject,
)
from .schema import Session as SessionRow

#: Legacy ``signal_types`` entries fold into the unified ``event_types`` table as data
#: events; map the canonical signal type to its neuroai ``Event`` subclass.
_SIGNAL_TYPE_TO_CLASS = {
    "threshold_crossings": "SpikeCounts",
    "spike_band_power": "BandPower",
    "sorted_spikes": "Spikes",
    "kinematics": "BehavioralSeries",
    "acoustic_features": "AcousticFeatures",
    "audio": "Audio",
    "synthesized_audio": "Audio",
    "lfp": "Ieeg",
    "ecog": "Ieeg",
    "emg": "Emg",
}

_EVENT_FIELDS = [
    "category",
    "event_class",
    "signal_type",
    "modality",
    "bin_size_ms",
    "rate_hz",
    "dtype",
    "task_condition",
    "description",
]

#: ``name`` keys the row; everything else must map to an ``event_types`` column.
_KNOWN_EVENT_KEYS = {"name", *_EVENT_FIELDS}


def _event_rows(meta: dict):
    """Yield unified event dicts: the merged ``events`` list, plus any legacy
    ``signal_types`` (folded as ``category="data"`` with an inferred ``event_class``).
    """
    for s in meta.get("signal_types", []):  # backward-compat for un-merged declarations
        yield {
            **s,
            "category": s.get("category", "data"),
            "event_class": s.get("event_class")
            or _SIGNAL_TYPE_TO_CLASS.get(s.get("signal_type")),
        }
    yield from meta.get("events", [])


PACKAGE_DIR = Path(__file__).resolve().parent.parent  # .../src/neuralshed
SRC_DIR = PACKAGE_DIR.parent  # .../src (import root)
DATASETS_DIR = PACKAGE_DIR / "datasets"


# --------------------------------------------------------------------------- #
# Loading per-dataset declarations
# --------------------------------------------------------------------------- #
def _load_metadata_module(name: str):
    """Import a dataset's ``metadata`` module (``neuralshed.datasets.<name>.metadata``)."""
    try:
        return importlib.import_module(f"neuralshed.datasets.{name}.metadata")
    except ModuleNotFoundError:
        return None


def discover_datasets() -> list[str]:
    """Return dataset folder names that have a ``metadata.py``."""
    return sorted(p.parent.name for p in DATASETS_DIR.glob("*/metadata.py"))


# --------------------------------------------------------------------------- #
# Upsert helpers
# --------------------------------------------------------------------------- #
def _get_or_create(sess: OrmSession, model, defaults: dict | None = None, **keys):
    """Return (obj, created): fetch by ``keys`` or create with ``keys`` + ``defaults``."""
    obj = sess.query(model).filter_by(**keys).one_or_none()
    if obj is None:
        obj = model(**keys, **(defaults or {}))
        sess.add(obj)
        return obj, True
    return obj, False


def _set_fields(obj, data: dict, fields: list[str]) -> None:
    for f in fields:
        if f in data:
            setattr(obj, f, data[f])


def _link_participant(
    sess: OrmSession, ds_id: str, label: str, participant_id: str | None
) -> str | None:
    """Link a subject to its cross-dataset ``Participant``.

    ``participant_id`` is the reference declared on the subject in ``metadata.py``; the
    participant record itself is defined in ``datasets/participants.py`` (the source of
    truth). Upserts the ``Participant`` row from that record and returns its id, or
    ``None`` (warned) when the subject declares no reference or references an id that isn't
    defined in the registry.
    """
    if not participant_id:
        print(
            f"  [participant] warning: subject ({ds_id!r}, {label!r}) declares no "
            f"'participant_id'; leaving it unlinked"
        )
        return None
    entry = participants.participant_by_id(participant_id)
    if entry is None:
        print(
            f"  [participant] warning: subject ({ds_id!r}, {label!r}) references unknown "
            f"participant_id {participant_id!r}; not in datasets/participants.py"
        )
        return None
    vocab.warn_unknown("handedness", entry.get("handedness"), vocab.HANDEDNESS)
    part, _ = _get_or_create(sess, Participant, id=participant_id)
    _set_fields(part, entry, list(participants.PARTICIPANT_FIELDS))
    sess.flush()
    return participant_id


def _brain_area_id(
    sess: OrmSession,
    subject_id: int,
    raw_label: str | None,
    brodmann=None,
    description=None,
    homba_id=None,
    uberon_id=None,
):
    if not raw_label:
        return None
    ba, _ = _get_or_create(sess, BrainArea, subject_id=subject_id, raw_label=raw_label)
    if brodmann and not ba.brodmann:
        ba.brodmann = brodmann
    if homba_id and not ba.homba_id:
        ba.homba_id = homba_id
    if uberon_id and not ba.uberon_id:
        ba.uberon_id = uberon_id
    if description and not ba.description:
        ba.description = description
    sess.flush()
    return ba.id


#: Region enrichment moved from arrays to the top-level ``brain_areas`` list; warn if
#: an array still inlines these so the value isn't silently dropped.
_ARRAY_BRAIN_AREA_FIELDS = ("homba_id", "uberon_id", "brodmann", "description")


def _check_unique_arrays(ds_id: str, subject: dict) -> None:
    """Raise when two arrays of a subject share hemisphere and name. SQL treats ``NULL`` as
    distinct in the unique constraint, so an unstated hemisphere is checked here."""
    seen = set()
    for a in subject.get("arrays", []):
        key = (a.get("hemisphere"), a["name"])
        if key in seen:
            raise ValueError(
                f"{ds_id}/{subject['label']}: two arrays named {a['name']!r} in hemisphere "
                f"{a.get('hemisphere')!r}"
            )
        seen.add(key)


def _warn_inline_brain_area_fields(a: dict) -> None:
    stray = [k for k in _ARRAY_BRAIN_AREA_FIELDS if k in a]
    if stray:
        print(
            f"  [vocab] warning: array {a.get('name')!r} inlines {stray}; "
            f"move these to the top-level 'brain_areas' entry for "
            f"{a.get('brain_area')!r}"
        )


#: Person-invariant subject keys
_PARTICIPANT_OWNED_FIELDS = ("handedness", "species", "sex", "implant_date")


def _warn_person_invariant_on_subject(ds_id: str, s: dict) -> None:
    stray = [k for k in _PARTICIPANT_OWNED_FIELDS if k in s]
    if stray:
        print(
            f"  [participant] warning: subject ({ds_id!r}, {s.get('label')!r}) declares "
            f"{stray}; these are person-invariant and are read from "
            f"datasets/participants.py instead"
        )


def _upsert_brain_areas(sess: OrmSession, subject_id: int, subject: dict) -> None:
    """Upsert a subject's ``brain_areas`` list (the BrainArea rows it owns). Each entry
    is one region (``raw_label`` + optional ``homba_id`` / ``uberon_id`` / ``brodmann`` /
    ``description``); the subject's arrays reference it by their ``brain_area`` ==
    ``raw_label`` string.
    """
    for ba in subject.get("brain_areas", []):
        _brain_area_id(
            sess,
            subject_id,
            ba.get("raw_label"),
            ba.get("brodmann"),
            ba.get("description"),
            homba_id=ba.get("homba_id"),
            uberon_id=ba.get("uberon_id"),
        )


_DATASET_FIELDS = [
    "title",
    "description",
    "source_data_platform",
    "source_id",
    "source_data_url",
    "status",
    "macro_task",
    "decode_target",
    "bibtex",
    "publication_doi",
    "code_repo",
    "licence",
    "notes",
    "study_class",
]


def upsert_catalog(sess: OrmSession, meta: dict) -> None:
    """Insert/update the catalog (dataset + children) from a DATASET_METADATA dict."""
    ds_id = meta["id"]
    # soft vocab checks
    vocab.warn_unknown(
        "source_data_platform",
        meta.get("source_data_platform"),
        vocab.SOURCE_DATA_PLATFORMS,
    )
    vocab.warn_unknown("macro_task", meta.get("macro_task"), vocab.MACRO_TASKS)
    vocab.warn_unknown("status", meta.get("status"), vocab.STATUSES)
    vocab.warn_unknown("decode_target", meta.get("decode_target"), vocab.DECODE_TARGETS)

    ds = sess.get(Dataset, ds_id) or Dataset(id=ds_id)
    _set_fields(ds, meta, _DATASET_FIELDS)
    sess.add(ds)
    sess.flush()

    if meta.get("brain_areas"):  # regions are subject-owned now
        print(
            "  [vocab] warning: top-level 'brain_areas' is ignored; declare each "
            "subject's regions in that subject's 'brain_areas' list"
        )

    for s in meta.get("subjects", []):
        _warn_person_invariant_on_subject(ds_id, s)
        subj, _ = _get_or_create(sess, Subject, dataset_id=ds_id, label=s["label"])
        _set_fields(subj, s, ["age", "time_since_implant", "clinical_condition"])
        subj.participant_id = _link_participant(
            sess, ds_id, s["label"], s.get("participant_id")
        )
        sess.flush()
        _upsert_brain_areas(sess, subj.id, s)  # this subject's BrainArea rows
        _check_unique_arrays(ds_id, s)
        for a in s.get("arrays", []):
            vocab.warn_unknown("device_type", a.get("device_type"), vocab.DEVICE_TYPES)
            vocab.warn_unknown("hemisphere", a.get("hemisphere"), vocab.HEMISPHERES)
            _warn_inline_brain_area_fields(a)
            arr, _ = _get_or_create(
                sess,
                Array,
                subject_id=subj.id,
                hemisphere=a.get("hemisphere"),
                name=a["name"],
            )
            _set_fields(
                arr,
                a,
                [
                    "device_type",
                    "n_channels",
                    "channel_start",
                    "channel_stop",
                    "stereotaxic_coordinates",
                ],
            )
            # Arrays reference one of their subject's regions by ``brain_area`` ==
            # BrainArea.raw_label; the region's ontology enrichment lives in the
            # subject's ``brain_areas`` list.
            arr.brain_area_id = _brain_area_id(sess, subj.id, a.get("brain_area"))
        sess.flush()

    for ev in _event_rows(meta):
        vocab.warn_unknown("category", ev.get("category"), vocab.EVENT_CATEGORIES)
        vocab.warn_unknown("event_class", ev.get("event_class"), vocab.EVENT_CLASSES)
        vocab.warn_unknown("signal_type", ev.get("signal_type"), vocab.SIGNAL_TYPES)
        vocab.warn_unknown("modality", ev.get("modality"), vocab.RECORDING_MODALITIES)
        vocab.warn_unknown(
            "task_condition", ev.get("task_condition"), vocab.TASK_CONDITIONS
        )
        vocab.warn_unknown_dtype(ev.get("dtype"))
        unknown = sorted(set(ev) - _KNOWN_EVENT_KEYS)
        if unknown:  # otherwise the key is silently dropped at _set_fields
            print(
                f"  [vocab] warning: event {ev.get('name')!r} declares {unknown}, "
                f"which have no column in event_types"
            )
        obj, _ = _get_or_create(sess, EventType, dataset_id=ds_id, name=ev["name"])
        _set_fields(obj, ev, _EVENT_FIELDS)

    # Per-session rows are folded into the dict by ``get_metadata()`` (empty without the
    # data drive). Subjects are inserted above, so ``subject_label`` linking resolves.
    upsert_sessions(sess, ds_id, meta.get("sessions", []))


def upsert_sessions(sess: OrmSession, ds_id: str, rows: list[dict]) -> int:
    """Insert/update Session rows for a dataset. Returns the count upserted."""
    subj_by_label = {
        s.label: s.id for s in sess.query(Subject).filter_by(dataset_id=ds_id)
    }
    n = 0
    for r in rows:
        vocab.warn_unknown(
            "task_condition", r.get("task_condition"), vocab.TASK_CONDITIONS
        )
        obj, _ = _get_or_create(sess, SessionRow, dataset_id=ds_id, label=r["label"])
        _set_fields(
            obj, r, ["date", "post_implant_day", "n_trials", "task", "task_condition"]
        )
        # tolerate session builders that still emit the old "split" key
        if "original_split" in r or "split" in r:
            obj.original_split = r.get("original_split", r.get("split"))
        if r.get("subject_label") in subj_by_label:
            obj.subject_id = subj_by_label[r["subject_label"]]
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def populate(
    db_url: str | None = None, catalog_only: bool = False, only: list[str] | None = None
) -> None:
    engine = db.get_engine(db_url) if db_url else db.get_engine()
    db.create_all(engine)
    names = only or discover_datasets()
    print(f"populating {len(names)} dataset(s) into {engine.url}")

    from sqlalchemy.orm import sessionmaker

    # Make `neuralshed` importable, and the datasets dir importable so each dataset's
    # `utils` can `import _neuralshed_paths` (shared data-root resolver) when the drive is up.
    for p in (SRC_DIR, DATASETS_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    Maker = sessionmaker(bind=engine, future=True)
    with Maker() as sess:
        for name in names:
            mod = _load_metadata_module(name)
            if mod is None or not hasattr(mod, "get_metadata"):
                print(f"  - {name}: no metadata.get_metadata(), skipped")
                continue
            meta = mod.get_metadata()
            n_sessions = len(meta.get("sessions", []))
            if catalog_only:
                meta = {k: v for k, v in meta.items() if k != "sessions"}
            upsert_catalog(sess, meta)
            if catalog_only:
                detail = " (catalog-only)"
            elif n_sessions:
                detail = f"; {n_sessions} sessions"
            else:
                detail = "; 0 sessions (drive not mounted)"
            print(f"  - {name}: catalog ok{detail}")
        sess.commit()
    print("done.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--db", help="database URL (default: env DATABASE_URL or local SQLite)"
    )
    p.add_argument(
        "--catalog-only", action="store_true", help="skip session-level population"
    )
    p.add_argument(
        "--datasets", help="comma-separated dataset names (default: all discovered)"
    )
    args = p.parse_args(argv)
    only = [s.strip() for s in args.datasets.split(",")] if args.datasets else None
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    populate(db_url=args.db, catalog_only=args.catalog_only, only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
