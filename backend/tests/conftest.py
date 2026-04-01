"""Shared test fixtures and configuration.

Test database strategy:
    We use an in-process SQLite database (via aiosqlite) for tests.
    This means tests run anywhere — no Docker, no PostgreSQL required.

    Why SQLite for tests instead of PostgreSQL?
    - Zero setup: tests run on any machine with Python
    - Speed: in-memory database, no network I/O
    - Isolation: each test session gets a fresh database
    - CI-friendly: no service dependencies in GitHub Actions

    Trade-offs we accept:
    - JSONB columns are treated as JSON (no containment queries)
    - pgvector features can't be tested (tested separately in integration)
    - PostgreSQL-specific CHECK constraints don't apply in SQLite

    For full integration tests (Phase 7), we'll add a separate conftest
    that spins up a real PostgreSQL container via testcontainers.

Session lifecycle:
    1. Create async SQLite engine (in-memory)
    2. Create all tables from ORM metadata
    3. Override FastAPI's get_db dependency to use test sessions
    4. Each test gets a fresh session with automatic rollback
    5. After all tests, engine is disposed
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.database import get_db
from app.core.models.db import Base
from app.main import create_app

# ── Test database engine (in-memory SQLite) ──────────────────────────────────
# Using SQLite for unit tests. PostgreSQL-specific features (JSONB operators,
# pgvector, CHECK constraints) are tested in integration tests with real PG.

_test_engine = create_async_engine(
    "sqlite+aiosqlite://",
    echo=False,
)

_test_session_factory = async_sessionmaker(
    bind=_test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Create all tables once per test session.

    Uses the same SQLAlchemy Base metadata as the real app, so the
    test database schema always matches the ORM models.
    """
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _test_engine.dispose()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a test database session with automatic cleanup.

    Each test gets its own session. After the test completes (success or
    failure), all changes are rolled back. This ensures test isolation —
    one test's data never leaks into another.
    """
    async with _test_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def app(db_session: AsyncSession):
    """Create a test application with database dependency overridden.

    The real get_db dependency connects to PostgreSQL. In tests, we
    override it to use our in-memory SQLite session instead.
    """
    application = create_app()

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    application.dependency_overrides[get_db] = _override_get_db
    return application


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP test client.

    Uses httpx's ASGITransport to call the FastAPI app directly
    without starting a real server — fast and reliable.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
