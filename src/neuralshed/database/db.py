"""Engine / session factory for the metadata database.

Backend is chosen by the ``DATABASE_URL`` environment variable:

- unset (default) -> a local SQLite file ``neuralshed/database/neuralshed_metadata.db``
- ``postgresql+psycopg://user:pass@host/dbname`` -> PostgreSQL via psycopg

The same ORM schema (``schema.py``) runs on either backend, so a first version
can develop on SQLite and move to Postgres later with no code change.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

#: Default on-disk SQLite location (next to this package).
DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent / "neuralshed_metadata.db"


def database_url() -> str:
    """Return the configured database URL (env ``DATABASE_URL`` or the SQLite default)."""
    return os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")


_engine: Engine | None = None
_Session: sessionmaker | None = None


def get_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Return a process-wide SQLAlchemy engine (created on first use).

    Parameters
    ----------
    url : str, optional
        Override the database URL (otherwise :func:`database_url`).
    echo : bool, optional
        Echo emitted SQL (debugging).
    """
    global _engine, _Session
    if url is not None:
        return create_engine(url, echo=echo, future=True)
    if _engine is None:
        _engine = create_engine(database_url(), echo=echo, future=True)
        _Session = sessionmaker(bind=_engine, future=True)
    return _engine


def get_session() -> Session:
    """Return a new ORM session bound to the default engine."""
    if _Session is None:
        get_engine()
    assert _Session is not None
    return _Session()


def create_all(engine: Engine | None = None) -> None:
    """Create all tables for the current schema (first-cut / tests).

    For versioned schema evolution use Alembic (``alembic upgrade head``).
    """
    from .schema import Base

    Base.metadata.create_all(engine or get_engine())
