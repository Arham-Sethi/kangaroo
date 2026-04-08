"""Tests for the analytics API endpoints.

Tests cover:
    - Daily usage endpoint structure
    - Model distribution endpoint
    - Summary endpoint
    - Authentication required
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.v1.analytics import router


# ── Test Helpers ──────────────────────────────────────────────────────────────


def _build_test_app() -> FastAPI:
    """Create a minimal FastAPI app for analytics testing."""
    app = FastAPI()
    app.include_router(router, prefix="/analytics")
    return app


def _mock_user():
    """Create a mock User object."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.subscription_tier = "pro"
    user.shifts_this_month = 5
    return user


def _mock_get_current_user(user: MagicMock):
    """Create a dependency override."""
    async def _override():
        return user
    return _override


def _mock_db():
    """Create a mock AsyncSession."""
    db = AsyncMock()

    # Default: empty result set for any execute call
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_result.__iter__ = lambda self: iter([])
    mock_result.scalar_one.return_value = 0
    db.execute.return_value = mock_result

    return db


def _mock_get_db(db: AsyncMock):
    async def _override():
        yield db
    return _override


@pytest.fixture
def user():
    return _mock_user()


@pytest.fixture
def db():
    return _mock_db()


@pytest.fixture
def app(user, db):
    from app.api.v1.auth import get_current_user
    from app.core.database import get_db

    test_app = _build_test_app()
    test_app.dependency_overrides[get_current_user] = _mock_get_current_user(user)
    test_app.dependency_overrides[get_db] = _mock_get_db(db)
    return test_app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDailyUsage:
    @pytest.mark.asyncio
    async def test_returns_list(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/daily-usage")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_default_30_days(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/daily-usage")
        data = resp.json()
        assert len(data) == 30

    @pytest.mark.asyncio
    async def test_custom_days(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/daily-usage?days=7")
        data = resp.json()
        assert len(data) == 7

    @pytest.mark.asyncio
    async def test_entry_structure(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/daily-usage?days=1")
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert "date" in entry
        assert "shifts" in entry
        assert "sessions" in entry
        assert "tokens" in entry

    @pytest.mark.asyncio
    async def test_rejects_invalid_days(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/daily-usage?days=0")
        assert resp.status_code == 422

        resp = await client.get("/analytics/daily-usage?days=100")
        assert resp.status_code == 422


class TestModelDistribution:
    @pytest.mark.asyncio
    async def test_returns_list(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/model-distribution")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestSummary:
    @pytest.mark.asyncio
    async def test_returns_summary(self, client: AsyncClient) -> None:
        resp = await client.get("/analytics/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_sessions" in data
        assert "total_shifts" in data
        assert "total_tokens" in data
        assert "shifts_this_month" in data
        assert "subscription_tier" in data

    @pytest.mark.asyncio
    async def test_summary_values(self, client: AsyncClient, user) -> None:
        resp = await client.get("/analytics/summary")
        data = resp.json()
        assert data["shifts_this_month"] == 5  # From mock user
        assert data["subscription_tier"] == "pro"
