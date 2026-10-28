"""Alembic migration environment for the neuralshed metadata database.

Wired to the project schema (``neuralshed.database.schema.Base.metadata``) and
the same ``DATABASE_URL`` resolution as the app (``neuralshed.database.db``), so
``alembic upgrade head`` builds/evolves the exact schema on SQLite or Postgres.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make `neuralshed` importable even when not pip-installed (src/ on sys.path).
SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neuralshed.database import db as _db  # noqa: E402
from neuralshed.database.schema import Base  # noqa: E402

config = context.config
# Resolve the URL the same way the app does (env DATABASE_URL or SQLite default).
config.set_main_option("sqlalchemy.url", _db.database_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,  # SQLite-safe ALTERs
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # SQLite-safe ALTERs
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
