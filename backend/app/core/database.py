"""Async database engine, session factory, and FastAPI dependency.

This module is the single source of truth for all database connectivity.
Every query in the entire backend flows through here. Design decisions:

    1. AsyncSession (not sync) — FastAPI is async, blocking DB calls would
       serialize all requests through the event loop, killing throughput.
       With async, one request waiting on Postgres doesn't block others.

    2. Connection pooling (pool_size=20, max_overflow=10) — Postgres has a
       hard limit on connections (~100 default). Pooling reuses connections
       across requests. 20 base + 10 overflow = handles 30 concurrent
       queries, which supports ~500 concurrent HTTP requests (most requests
       are not in a DB query at any given moment).

    3. Pool pre-ping — Before handing out a connection, the pool checks if
       it's still alive. This prevents "connection reset" errors after
       Postgres restarts or network blips. Costs ~1ms per checkout.

    4. Expire-on-commit disabled — After committing, SQLAlchemy normally
       invalidates all loaded objects (forcing a re-query on next access).
       We disable this because our API pattern is: commit → serialize to
       response. We need the objects to be readable after commit.

    5. Transaction-per-request — Each HTTP request gets its own session.
       If the request succeeds → commit. If it raises → rollback. This
       ensures atomicity: a request either fully succeeds or fully fails.
       No partial writes that leave the database in an inconsistent state.

Usage in routes:
    from app.core.database import get_db

    @router.get("/items")
    async def list_items(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Item))
        return result.scalars().all()
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

# ── Module-level engine (initialized on first import) ────────────────────────
# The engine is created once and shared across the entire application process.
# Each uvicorn worker gets its own engine with its own connection pool.

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine.

    The engine manages the connection pool. It is:
    - Created once per process (cached in module global)
    - Thread-safe and async-safe
    - Configured from environment variables via Settings

    Pool configuration rationale:
    - pool_size=20: Base connections kept alive. Handles normal load.
    - max_overflow=10: Extra connections under spike. Closed after use.
    - pool_recycle=3600: Recreate connections after 1 hour to prevent
      stale connections from sitting in the pool after Postgres maintenance.
    - pool_pre_ping=True: Verify connection health before use. Prevents
      "connection already closed" errors after network interruptions.
    - echo=False: SQL logging disabled in production (use debug=True in dev).
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_recycle=settings.db_pool_recycle,
            pool_pre_ping=True,
            echo=settings.debug,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory.

    The factory creates new sessions with consistent configuration:
    - expire_on_commit=False: Objects remain readable after commit
      (needed for serializing response data after the commit).
    - autocommit/autoflush disabled: Explicit transaction control.
      We commit at the end of the request, not after every operation.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides a database session per request.

    Usage pattern (injected into every route that needs DB access):

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Lifecycle:
        1. Request arrives → session created from pool
        2. Route handler executes queries
        3. If no exception → session.commit() (persist changes)
        4. If exception → session.rollback() (discard changes)
        5. Session always closed → connection returned to pool

    This guarantees:
        - Every request is atomic (all-or-nothing)
        - Connections are always returned to the pool
        - No manual commit/rollback needed in route handlers
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize the database engine and verify connectivity.

    Called during application startup. If the database is unreachable,
    the app fails fast with a clear error rather than accepting requests
    and failing on the first DB query.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        # Simple connectivity check — will raise if DB is unreachable
        await conn.execute(
            __import__("sqlalchemy").text("SELECT 1")
        )


async def close_db() -> None:
    """Close the database engine and all pooled connections.

    Called during application shutdown. Ensures all connections are
    properly closed and returned to PostgreSQL.
    """
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
