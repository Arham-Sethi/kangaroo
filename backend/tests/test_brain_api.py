"""Tests for brain API endpoints — digest, gaps, and stats.

Tests cover:
    - POST /api/v1/brain/digest — digest generation
    - POST /api/v1/brain/gaps — gap detection
    - GET /api/v1/brain/stats — brain statistics
    - Authentication required
    - Empty results
    - User isolation
"""

import uuid

import pytest


# -- Helpers -----------------------------------------------------------------


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


async def _register_and_get_token(client, email: str | None = None):
    if email is None:
        email = _unique_email()
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"]


async def _create_session(client, token, **overrides):
    body = {
        "title": "Test Session",
        "source_llm": "openai",
        "source_model": "gpt-4o",
        "message_count": 10,
        "total_tokens": 1500,
        "tags": ["test"],
    }
    body.update(overrides)
    resp = await client.post(
        "/api/v1/sessions",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    return resp.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# -- Digest Tests -----------------------------------------------------------


class TestBrainDigest:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/brain/digest",
            json={"max_age_days": 7},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_digest(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/brain/digest",
            json={"max_age_days": 7, "period_label": "Last week"},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_count"] == 0
        assert data["entries"] == []
        assert data["period_label"] == "Last week"

    @pytest.mark.asyncio
    async def test_digest_with_sessions(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(client, token, title="Python API Design", message_count=15)
        await _create_session(client, token, title="Database Schema", message_count=20)

        resp = await client.post(
            "/api/v1/brain/digest",
            json={"max_age_days": 30, "period_label": "This month"},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_count"] == 2
        assert data["total_messages"] == 35
        assert data["period_label"] == "This month"

    @pytest.mark.asyncio
    async def test_digest_default_values(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/brain/digest",
            json={},
            headers=_auth(token),
        )
        assert resp.status_code == 200


# -- Gaps Tests --------------------------------------------------------------


class TestBrainGaps:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/brain/gaps",
            json={},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_gaps(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/brain/gaps",
            json={"min_mentions": 2, "stalled_days": 14},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions_analyzed"] == 0
        assert data["gaps"] == []

    @pytest.mark.asyncio
    async def test_gaps_with_sessions(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(client, token, title="Session 1")
        await _create_session(client, token, title="Session 2")

        resp = await client.post(
            "/api/v1/brain/gaps",
            json={"min_mentions": 2, "stalled_days": 14},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions_analyzed"] == 2

    @pytest.mark.asyncio
    async def test_gaps_default_values(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/brain/gaps",
            json={},
            headers=_auth(token),
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_gaps_validation(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/brain/gaps",
            json={"min_mentions": 0},  # Must be >= 1
            headers=_auth(token),
        )
        assert resp.status_code == 422


# -- Stats Tests -------------------------------------------------------------


class TestBrainStats:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.get("/api/v1/brain/stats")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_stats(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get(
            "/api/v1/brain/stats",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 0
        assert data["total_contexts"] == 0
        assert data["total_messages"] == 0
        assert data["total_tokens"] == 0
        assert data["active_sessions"] == 0
        assert data["archived_sessions"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_sessions(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(client, token, message_count=10, total_tokens=500)
        await _create_session(client, token, message_count=20, total_tokens=1000)

        resp = await client.get(
            "/api/v1/brain/stats",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 2
        assert data["active_sessions"] == 2
        assert data["archived_sessions"] == 0
        assert data["total_messages"] == 30
        assert data["total_tokens"] == 1500

    @pytest.mark.asyncio
    async def test_stats_user_isolation(self, client) -> None:
        token_a = await _register_and_get_token(client)
        token_b = await _register_and_get_token(client)

        await _create_session(client, token_a, message_count=100)
        await _create_session(client, token_a, message_count=200)

        resp = await client.get(
            "/api/v1/brain/stats",
            headers=_auth(token_b),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 0
        assert data["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_archived(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        # Archive the session
        await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth(token),
        )

        await _create_session(client, token)  # Active session

        resp = await client.get(
            "/api/v1/brain/stats",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 2
        assert data["active_sessions"] == 1
        assert data["archived_sessions"] == 1
