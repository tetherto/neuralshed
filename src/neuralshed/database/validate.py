"""Soft validation of per-dataset declarations against the DB schema.

No running database required: each dataset's ``get_metadata()`` dict is populated into a
throwaway **in-memory SQLite** built from the current ORM schema (``schema.Base``). A
dataset *validates* if it ingests without error; controlled-vocabulary mismatches are
reported as soft **warnings** (they don't fail validation). Use it as a check whenever a
dataset's metadata changes — before the real database is even running.

    python -m neuralshed.database.validate                      # all datasets
    python -m neuralshed.database.validate --datasets card_2024  # a subset
    python -m neuralshed.database.validate --strict             # warnings count as failures

Programmatic::

    from neuralshed.database.validate import validate_dataset, validate_metadata
    res = validate_dataset("wairagkar_2025")          # by name (imports its metadata)
    res = validate_metadata(some_meta_dict)           # or validate a dict directly
    res.ok, res.warnings, res.counts
"""

import argparse
import contextlib
import io
import sys
from dataclasses import dataclass, field

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from .populate import _load_metadata_module, discover_datasets, upsert_catalog
from .schema import Base

#: Tables a dataset's ``get_metadata()`` fills (``sessions`` is 0 without the data drive).
_CATALOG_TABLES = (
    "datasets",
    "subjects",
    "brain_areas",
    "arrays",
    "event_types",
    "sessions",
)


@dataclass
class Result:
    """Outcome of validating one dataset against the schema."""

    name: str
    ok: bool
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def __bool__(self) -> bool:  # truthy iff it validated
        return self.ok


def validate_metadata(meta: dict, name: str | None = None) -> Result:
    """Populate one ``get_metadata()`` dict into a throwaway in-memory SQLite and report.

    Returns a :class:`Result`; ``ok`` is False only on a hard schema error (missing
    required field, unique/constraint violation, bad type the backend rejects). Unknown
    controlled-vocabulary values are collected in ``warnings``.
    """
    name = name or meta.get("id", "?")
    engine = sa.create_engine("sqlite://", future=True)  # in-memory, throwaway
    sa.event.listen(  # enforce FKs so dangling references surface
        engine, "connect", lambda dbapi, _r: dbapi.execute("PRAGMA foreign_keys=ON")
    )
    Base.metadata.create_all(engine)
    Maker = sessionmaker(bind=engine, future=True)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):  # capture vocab.warn_unknown output
            with Maker() as sess:
                upsert_catalog(sess, meta)
                sess.commit()
                counts = {
                    t: sess.execute(sa.text(f"SELECT COUNT(*) FROM {t}")).scalar()
                    for t in _CATALOG_TABLES
                }
    except Exception as e:
        warns = _warnings(buf)
        return Result(name, ok=False, error=f"{type(e).__name__}: {e}", warnings=warns)
    return Result(name, ok=True, warnings=_warnings(buf), counts=counts)


def validate_dataset(name: str) -> Result:
    """Validate a discovered dataset: import its ``metadata`` and run ``get_metadata()``."""
    mod = _load_metadata_module(name)
    if mod is None or not hasattr(mod, "get_metadata"):
        return Result(name, ok=False, error="no metadata.get_metadata()")
    # get_metadata() -> sessions() does a bare `import utils`; evict cached siblings.
    for _m in ("utils", "source_data_download"):
        sys.modules.pop(_m, None)
    return validate_metadata(mod.get_metadata(), name=name)


def _warnings(buf: io.StringIO) -> list[str]:
    return [ln.strip() for ln in buf.getvalue().splitlines() if "[vocab]" in ln]


def validate_all(names: list[str] | None = None) -> list[Result]:
    """Validate every discovered dataset (or a given subset)."""
    return [validate_dataset(n) for n in (names or discover_datasets())]


# --------------------------------------------------------------------------- #
# Session-label cross-check (drive/network dependent — opt-in)
# --------------------------------------------------------------------------- #
@dataclass
class SessionCheck:
    """Whether a Study's timeline ``session`` ids all match ``sessions.label`` rows."""

    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str = ""
    n_timelines: int | None = None
    n_labels: int | None = None
    missing: list[str] = field(default_factory=list)


def _timeline_sessions(name: str) -> tuple[set[str] | None, int]:
    """``({timeline["session"]}, n_timelines)`` from the dataset's Study.

    ``(None, 0)`` when the dataset has no ``study.py`` Study subclass. Needs the data
    drive / network to enumerate timelines; raises otherwise (caller turns it into a skip).
    """
    import importlib
    import inspect

    from neuralset.events import study as _study_base

    from neuralshed.datasets.study import NeuralshedStudy
    from neuralshed.datasets.utils import dataset_dir

    try:
        mod = importlib.import_module(f"neuralshed.datasets.{name}.study")
    except ModuleNotFoundError:
        return None, 0
    cls = next(
        (
            obj
            for obj in vars(mod).values()
            if inspect.isclass(obj)
            and issubclass(obj, _study_base.Study)
            and obj not in (_study_base.Study, NeuralshedStudy)
        ),
        None,
    )
    if cls is None:
        return None, 0
    # Data is staged at <root>/<name>/data/<StudyClassName>/; the Study base appends the
    # class-name subfolder, so hand it the `data/` dir
    study = cls(path=dataset_dir(name) / "data")
    ids, n = set(), 0
    for tl in study.iter_timelines():
        ids.add(tl["session"])
        n += 1
    return ids, n


def check_session_labels(name: str) -> SessionCheck:
    """Assert every timeline ``session`` has a matching ``sessions.label`` row.

    This is the join key for ``DB-select ∘ Study.iter_timelines``. Both sides need the
    data (Study enumeration + ``sessions()``), so any unavailability is a **skip**, never a
    fail — run it where the data lives.
    """
    meta_mod = _load_metadata_module(name)
    if meta_mod is None or not hasattr(meta_mod, "sessions"):
        return SessionCheck(name, "skip", "no metadata.sessions()")
    for _m in ("utils", "source_data_download"):  # evict cached siblings (bare imports)
        sys.modules.pop(_m, None)
    try:
        labels = {r["label"] for r in meta_mod.sessions()}
    except Exception as e:  # drive/network missing, etc.
        return SessionCheck(name, "skip", f"sessions() unavailable: {type(e).__name__}")
    if not labels:
        return SessionCheck(name, "skip", "sessions() empty (data drive not mounted?)")

    try:
        tl_sessions, n_tl = _timeline_sessions(name)
    except Exception as e:
        return SessionCheck(name, "skip", f"study unavailable: {type(e).__name__}")
    if tl_sessions is None:
        return SessionCheck(name, "skip", "no study.py Study subclass")
    if (
        n_tl == 0
    ):  # study reached no data (its download root differs from the staged path)
        return SessionCheck(
            name, "skip", "study yielded 0 timelines (data not reachable)"
        )

    missing = sorted(tl_sessions - labels)
    status = "fail" if missing else "ok"
    return SessionCheck(
        name,
        status,
        detail="timeline session(s) not in sessions.label" if missing else "",
        n_timelines=n_tl,
        n_labels=len(labels),
        missing=missing,
    )


def check_all_session_labels(names: list[str] | None = None) -> list[SessionCheck]:
    """Run :func:`check_session_labels` over every discovered dataset (or a subset)."""
    return [check_session_labels(n) for n in (names or discover_datasets())]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", help="comma-separated dataset names (default: all)")
    p.add_argument(
        "--strict", action="store_true", help="treat vocab warnings as failures"
    )
    p.add_argument(
        "--check-sessions",
        action="store_true",
        help="cross-check timeline session ids against sessions.label "
        "(needs the data drive/network; datasets without data are skipped)",
    )
    args = p.parse_args(argv)
    only = [s.strip() for s in args.datasets.split(",")] if args.datasets else None

    if args.check_sessions:
        return _main_check_sessions(only)

    results = validate_all(only)
    failed = 0
    for r in results:
        bad = (not r.ok) or bool(args.strict and r.warnings)
        failed += bad
        mark = "FAIL" if bad else "ok"
        extra = f" ({len(r.warnings)} warning(s))" if r.warnings else ""
        print(f"  [{mark}] {r.name}{extra}")
        if r.error:
            print(f"        error: {r.error}")
        for w in r.warnings:
            print(f"        {w}")
    print(
        f"\n{len(results) - failed}/{len(results)} datasets validate"
        + (" (strict)" if args.strict else "")
    )
    return 1 if failed else 0


def _main_check_sessions(only: list[str] | None) -> int:
    """``--check-sessions`` entry point: report ok/fail/skip per dataset, fail on mismatch."""
    checks = check_all_session_labels(only)
    failed = 0
    for c in checks:
        failed += c.status == "fail"
        counts = (
            f" ({c.n_timelines} timelines / {c.n_labels} labels)"
            if c.n_timelines is not None
            else ""
        )
        note = f" — {c.detail}" if c.detail else ""
        print(f"  [{c.status}] {c.name}{counts}{note}")
        for m in c.missing[:10]:
            print(f"        missing label for session: {m!r}")
        if len(c.missing) > 10:
            print(f"        … and {len(c.missing) - 10} more")
    n_ok = sum(c.status == "ok" for c in checks)
    n_skip = sum(c.status == "skip" for c in checks)
    print(f"\n{n_ok} ok, {failed} fail, {n_skip} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
