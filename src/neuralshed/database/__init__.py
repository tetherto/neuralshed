"""neuralshed metadata database — a searchable index over the curated dataset collection.

First version (SOW §3.2): SQLAlchemy ORM, SQLite by default and PostgreSQL-ready
via the ``DATABASE_URL`` environment variable. See ``README.md``.

Typical use::

    from neuralshed.database import query
    query.datasets(signal_type="spike_band_power")
    query.sessions(dataset="card_2024")

Rebuild the catalog from the per-dataset declarations::

    python -m neuralshed.database.populate            # catalog + sessions (if drive mounted)
    python -m neuralshed.database.populate --catalog-only
"""

from . import collections, db, query, schema, vocab  # noqa: F401

# ``populate`` and ``validate`` are CLI entry points (``python -m ...``); importing them
# here would trigger a runpy double-import warning, so they're left as plain submodules.
__all__ = ["db", "schema", "vocab", "query", "collections"]
