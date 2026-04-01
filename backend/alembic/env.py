"""Alembic environment configuration.

This connects Alembic to our SQLAlchemy models so it can:
    - Auto-detect schema changes (alembic revision --autogenerate)
    - Generate migration files with upgrade/downgrade SQL
    - Run migrations against any environment (dev, staging, prod)

The database URL is read from environment variables via our config
system — never hardcoded. This ensures migrations work in Docker,
CI/CD pipelines, and production without code changes.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.models.db import Base

# Alembic Config object — provides access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the MetaData that Alembic uses for --autogenerate.
# It must reference the same Base that all our models inherit from.
target_metadata = Base.metadata


def get_database_url() -> str:
    """Get the sync database URL for migrations.

    Priority: env var > alembic.ini setting.
    Migrations always use sync driver (not asyncpg).
    """
    return os.getenv(
        "DATABASE_SYNC_URL",
        config.get_main_option("sqlalchemy.url", ""),
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without connecting.

    Useful for generating migration scripts to review before applying.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — connects and applies changes.

    Uses NullPool because migrations are short-lived operations
    that don't benefit from connection pooling.
    """
    connectable = create_engine(
        get_database_url(),
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
