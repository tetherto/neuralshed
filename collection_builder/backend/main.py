import contextlib
import sys
import tempfile
import typing as tp
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))  # so `import catalog` resolves regardless of CWD

import neuralset as ns  # noqa: E402

import catalog as cat  # noqa: E402
import neuralshed  # noqa: E402
import neuralshed.studies  # noqa: E402,F401 — registers every dataset's Study
import neuralshed.transforms  # noqa: E402,F401 — registers CombineStudies/DummyTransform
from neuralshed.database import collections  # noqa: E402
from neuralshed.database.schema import COLLECTION_NAME_RE  # noqa: E402

DIST_DIR = APP_DIR / "frontend" / "dist"
DATASETS_DIR = Path(neuralshed.__file__).resolve().parent / "datasets"

app = FastAPI(title="Collection Builder API")

# Dev: the Vite server proxies /api, so this is only a fallback for running it unproxied.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- catalog cache ---------------------------------------------------------
# The catalog (datasets + region tree) is static within a run; saved collections are live.
_CATALOG: list[dict] | None = None


def catalog() -> list[dict]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = cat.load_catalog()
    return _CATALOG


def by_id() -> dict[str, dict]:
    return {d["id"]: d for d in catalog()}


# --- steps -----------------------------------------------------------------
# A study and a transform are both neuroai Steps, so the builder renders one card for both,
# driven by the params derived here from each step's pydantic model. Nothing is hand-kept.

#: Which registered EventsTransforms the builder offers. Grow as they are vetted for iBCI
#: data; the registry itself holds ~two dozen, many MEG/EEG-specific.
TRANSFORM_ALLOWLIST = (
    "DummyTransform",
    "StandardizeText",
    "ValidateTarget",
    "AssignHashSplit",  # splits on a composite key, so a trial can be finer than a timeline
    "SklearnSplit",  # neuralset's random train/val/test split by a grouping column
)

#: Execution/wiring knobs, not part of what a step *means*; never shown as a parameter.
#: ``path`` is environment (injected at load), so a stored config never carries it.
#: ``version`` is neuralset's cache-busting key, bumped when a Study's loading code changes.
_HIDDEN_PARAMS = {"infra", "path", "timelines", "version"}


def _params(cls: type) -> dict:
    schema = cls.model_json_schema()
    return {
        k: v for k, v in schema.get("properties", {}).items() if k not in _HIDDEN_PARAMS
    }


def _step(cls: type) -> dict:
    """A step's UI contract: its name, one-line doc, and its editable params."""
    return {
        "name": cls.__name__,
        "description": (cls.__doc__ or "").strip().split("\n")[0],
        "params": _params(cls),
    }


def transforms() -> list[dict]:
    registry = ns.EventsTransform._get_discriminated_subclasses()
    return [_step(registry[n]) for n in TRANSFORM_ALLOWLIST if n in registry]


def study_params() -> dict:
    """Editable params shared by every ``Study`` step (``query``)."""
    return _params(ns.Study)


def event_class_kinds(cat_list: list[dict]) -> list[dict]:
    """Every event class in the catalog, tagged ``data`` / ``text`` / ``categorical``.

    The kind comes from the neuroai class hierarchy rather than the ``event_types.category``
    column, so it is the same answer the manifest will give. It decides which classes the UI
    offers as inputs (dense arrays) and which as targets.
    """
    from neuralshed.manifest import kind_of

    names = sorted({c for d in cat_list for c in d["event_classes"]})
    out = []
    for name in names:
        try:
            kind = kind_of(name)
        except KeyError:  # declared in the catalog, no Event subclass registered for it
            kind = None
        out.append({"name": name, "kind": kind})
    return out


# --- validating a composed chain -------------------------------------------
# The blob is client-supplied and ends up as `ns.Step(**chain)`, which instantiates classes by
# name. Vet it here, not just in the UI.


def _member_steps(chain: dict) -> tuple[list[dict], list[dict]]:
    """A chain's member heads (the studies) and every other step in it."""
    try:
        steps = chain["steps"]
        chains = steps["CombineStudies"]["chains"]
        rest = [s for k, s in steps.items() if k != "CombineStudies"]
        heads = []
        for member in chains.values():
            member = member if isinstance(member, list) else [member]
            heads.append(member[0])
            rest.extend(member[1:])
    except (AttributeError, IndexError, KeyError, TypeError) as e:
        raise HTTPException(
            400, "config must be a Chain headed by CombineStudies"
        ) from e
    if not heads or not all(
        isinstance(s, dict) and isinstance(s.get("name"), str) for s in heads + rest
    ):
        raise HTTPException(400, "every step must be an object with a `name`")
    return heads, rest


def _reserved_keys(obj: tp.Any) -> set[str]:
    if isinstance(obj, dict):
        return _HIDDEN_PARAMS & set(obj) | set().union(
            *(_reserved_keys(v) for v in obj.values()), set()
        )
    if isinstance(obj, list):
        return set().union(*(_reserved_keys(v) for v in obj), set())
    return set()


@contextlib.contextmanager
def _preserved_study_paths() -> tp.Iterator[None]:
    # Constructing a Study records its path in a module global that SpecialLoader reads back
    # later; validating against a temp dir would leave a dangling path there.
    from neuralset.events.study import STUDY_PATHS

    snapshot = dict(STUDY_PATHS)
    try:
        yield
    finally:
        STUDY_PATHS.clear()
        STUDY_PATHS.update(snapshot)


def _validate_chain(chain: dict, sel: list[dict]) -> None:
    """Reject a chain that would not load, before it reaches the database."""
    heads, rest = _member_steps(chain)

    if bad := _reserved_keys(chain):
        raise HTTPException(400, f"config must not set {sorted(bad)}: injected at load")

    want = {d["study_class"] for d in sel}
    if (got := {s["name"] for s in heads}) != want:
        raise HTTPException(
            400, f"members {sorted(got)} do not match the selection {sorted(want)}"
        )

    if bad := sorted({s["name"] for s in rest} - set(TRANSFORM_ALLOWLIST)):
        raise HTTPException(
            400, f"transform not offered by this builder: {', '.join(bad)}"
        )

    # Paths are the one thing a stored config lacks, so borrow a throwaway root to type-check it.
    with tempfile.TemporaryDirectory() as tmp, _preserved_study_paths():
        try:
            ns.Step(**collections.inject_paths(chain, dict.fromkeys(want, tmp)))
        except (ImportError, KeyError, ValidationError, ValueError) as e:
            raise HTTPException(
                400, f"config does not load: {str(e).splitlines()[0]}"
            ) from e


# --- serializers -----------------------------------------------------------
def _dataset(d: dict) -> dict:
    return {
        "id": d["id"],
        "title": d["title"],
        "study_class": d["study_class"],
        "event_classes": d["event_classes"],
        "decode_target": d["decode_target"],
        "region_lca": d["region_lca"],
        "region_label": d["region_label"],
        "region_short": cat.region_short_label(d["region_lca"]),
        "channels": d["channels"],
        "rates": d["rates"],
        "signals": [
            {"type": s, "badge": cat.signal_badge(s), "primary": s in cat.SPIKE_SIGNALS}
            for s in d["signals"]
        ],
        "participants": d["participants"],
        "participants_short": [p.split("/")[-1] for p in d["participants"]],
        "subjects": d["subjects"],
        "has_sbp": d["has_sbp"],
        "has_primary": d["has_primary"],
        "platform": d["platform"],
        "under": sorted(d["under"]),
    }


def _region_tree(cat_list: list[dict]) -> dict:
    """Nodes reachable from the display roots, each with ordered children (busiest first) and
    rolled-up per-target counts, so the frontend can render the tree without more round-trips.
    """

    def ordered(nid: str) -> list[str]:
        return sorted(
            cat.REGION_CHILDREN.get(nid, []),
            key=lambda k: (-len(cat.datasets_under(k, cat_list)), cat.region_label(k)),
        )

    nodes: dict[str, dict] = {}

    def walk(nid: str) -> None:
        if nid in nodes:
            return
        kids = ordered(nid)
        nodes[nid] = {
            "id": nid,
            "label": cat.region_label(nid),
            "short": cat.region_short_label(nid),
            "parent": cat._NODES.get(nid, {}).get("parent"),
            "children": kids,
            "counts": cat.target_counts(nid, cat_list),
            "n_datasets": len(cat.datasets_under(nid, cat_list)),
        }
        for k in kids:
            walk(k)

    for r in cat.DISPLAY_ROOTS:
        walk(r)
    return {"display_roots": cat.DISPLAY_ROOTS, "nodes": nodes}


def _saved() -> list[dict]:
    """Saved collections, each with the chain and the declared streams it was composed with, so
    loading one restores it rather than re-saving defaults over the stored config."""
    out = []
    for _, r in collections.list_collections().iterrows():
        c = collections.get_collection(collection_name=r["name"])
        if c is None:  # deleted between the two reads
            continue
        out.append(
            {
                "id": int(r["id"]),
                "name": c["name"],
                "n_datasets": len(c["datasets"]),
                "datasets": sorted(c["datasets"]),
                "chain": c["chain"],
                "input_event_types": c["input_event_types"],
                "target_event_types": c["target_event_types"],
                "manifest": collections.manifest_status(collection_name=c["name"]),
            }
        )
    return out


def _streams(ids: list[str], wanted_in: list[str], wanted_out: list[str]) -> dict:
    """Which members can supply the declared streams, phrased for the UI.

    Reads the catalog only, so it answers while somebody is still assembling a collection, with
    no data drive mounted. Missing *targets* block a save (that member contributes no trials);
    missing *inputs* only warn, since inputs are wanted-if-present.
    """
    report = collections.check_declared_streams(
        dataset_ids=ids,
        input_event_types=wanted_in,
        target_event_types=wanted_out,
    )
    verdict = "ok"
    if report["missing_targets"]:
        verdict = "block"
    elif report["missing_inputs"] or any(
        set(wanted_in) - set(found) for found in report["inputs"].values()
    ):
        verdict = "warn"
    return {**report, "verdict": verdict}


def _analysis(ids: list[str]) -> dict:
    a = cat.analyze(ids, by_id())
    return {
        "verdict": a["verdict"],
        "rows": [{"status": s, "key": k, "value": v} for s, k, v in a["rows"]],
        "targets": a["targets"],
        "rates": a["rates"],
        "parts": a["parts"],
        "parts_short": [p.split("/")[-1] for p in a["parts"]],
        "adapters": len(a["sel"]),
        "n_subjects": len(a["parts"]),
    }


# --- request bodies --------------------------------------------------------
class AnalyzeBody(BaseModel):
    ids: list[str]


class StreamsBody(BaseModel):
    ids: list[str]
    input_event_types: list[str] = []
    target_event_types: list[str] = []


class SaveBody(BaseModel):
    name: str
    ids: list[str]
    chain: dict | None = None  # composed step chain (Compose tab); None until composed
    # declared intent: neuroai event class names. Inputs wanted-if-present, targets required.
    input_event_types: list[str] = []
    target_event_types: list[str] = []


# --- API -------------------------------------------------------------------
@app.get("/api/bootstrap")
def bootstrap() -> dict:
    """Everything the app needs on load: datasets, display vocab, region tree, suggestions,
    saved collections."""
    cat_list = catalog()
    return {
        "target_order": cat.TARGET_ORDER,
        "target_meta": cat.TARGET_META,
        "datasets": [_dataset(d) for d in cat_list],
        "regions": _region_tree(cat_list),
        "suggestions": cat.suggestions(cat_list),
        "saved": _saved(),
        "transforms": transforms(),
        "study_params": study_params(),
        "event_classes": event_class_kinds(cat_list),
    }


@app.get("/api/transforms")
def list_transforms() -> list[dict]:
    """Transforms the builder can chain, per study or per collection."""
    return transforms()


@app.post("/api/analyze")
def analyze(body: AnalyzeBody) -> dict:
    """Compatibility of a chosen set, phrased as model consequences."""
    return _analysis(body.ids)


@app.post("/api/streams")
def streams(body: StreamsBody) -> dict:
    """Which members can supply the declared inputs and targets, from the catalog alone."""
    unknown = sorted(set(body.ids) - set(by_id()))
    if unknown:
        raise HTTPException(400, f"unknown dataset(s): {', '.join(unknown)}")
    return _streams(body.ids, body.input_event_types, body.target_event_types)


@app.get("/api/collections")
def list_saved() -> list[dict]:
    return _saved()


@app.post("/api/collections")
def save(body: SaveBody) -> dict:
    """Persist the selection + its composed step chain."""
    name = body.name.strip()
    if not COLLECTION_NAME_RE.fullmatch(name):
        raise HTTPException(
            400, "name must be non-empty: letters, digits, '-' and '_' only"
        )
    catalog_by_id = by_id()
    if not body.ids:
        raise HTTPException(400, "no datasets in selection")
    if len(set(body.ids)) != len(body.ids):
        raise HTTPException(400, "duplicate datasets in selection")
    if unknown := sorted(set(body.ids) - set(catalog_by_id)):
        raise HTTPException(400, f"unknown dataset(s): {', '.join(unknown)}")
    sel = [catalog_by_id[i] for i in body.ids]
    # The catalog's per-dataset decode_target still gates poolability, and it answers before
    # anything is declared. The finer check is check_declared_streams, inside save_collection.
    if len({d["decode_target"] for d in sel}) != 1:
        raise HTTPException(400, "selection is not poolable (mixed decode targets)")
    if body.chain is not None:
        _validate_chain(body.chain, sel)
    try:
        collections.save_collection(
            collection_name=name,
            members=[d["id"] for d in sel],
            chain=body.chain,
            input_event_types=body.input_event_types,
            target_event_types=body.target_event_types,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e).strip("'")) from e
    return {"saved": name, "collections": _saved()}


@app.delete("/api/collections/{name}")
def remove(name: str) -> dict:
    deleted = collections.delete_collection(collection_name=name)
    return {"deleted": deleted, "collections": _saved()}


@app.get("/api/datasets/{dataset_id}/readme")
def readme(dataset_id: str) -> dict:
    """Raw README.md markdown for a dataset, read from its package directory."""
    if dataset_id not in by_id():  # also guards against path traversal
        raise HTTPException(404, "unknown dataset")
    path = DATASETS_DIR / dataset_id / "README.md"
    if not path.exists():
        raise HTTPException(404, "no README for this dataset")
    return {"id": dataset_id, "markdown": path.read_text()}


# --- static frontend (production) ------------------------------------------
# Mounted last so the /api routes above win; only present after `npm run build`.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")
