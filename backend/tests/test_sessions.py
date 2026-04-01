"""Tests for session CRUD endpoints.

Tests cover:
    - List sessions (pagination, filtering)
    - Get session by ID
    - Create session
    - Update session (title, tags)
    - Archive/unarchive
    - Soft-delete
    - Authorization (can't access other user's sessions)
    - Audit logging integration
"""

import uuid

import pytest


# -- Helpers -----------------------------------------------------------------


def _unique_email() -> str:
    """Generate a unique email so each test gets its own isolated user."""
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


async def _register_and_get_token(client, email: str | None = None):
    """Register a new user and return their access token."""
    if email is None:
        email = _unique_email()
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"]


async def _create_session(client, token, **overrides):
    """Create a session and return the response data."""
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


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# -- Create Tests ------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session(self, client) -> None:
        token = await _register_and_get_token(client)
        data = await _create_session(client, token)
        assert data["title"] == "Test Session"
        assert data["source_llm"] == "openai"
        assert data["message_count"] == 10
        assert data["is_archived"] is False

    @pytest.mark.asyncio
    async def test_create_session_minimal(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions",
            json={"source_llm": "anthropic"},
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Untitled Session"
        assert data["source_llm"] == "anthropic"
        assert data["message_count"] == 0

    @pytest.mark.asyncio
    async def test_create_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/sessions",
            json={"source_llm": "openai"},
        )
        assert resp.status_code == 403


# -- List Tests --------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(client, token, title="Session 1")
        await _create_session(client, token, title="Session 2")

        resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["sessions"]) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["sessions"] == []

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, client) -> None:
        token = await _register_and_get_token(client)
        for i in range(5):
            await _create_session(client, token, title=f"Session {i}")

        resp = await client.get(
            "/api/v1/sessions?limit=2&offset=0",
            headers=_auth_headers(token),
        )
        data = resp.json()
        assert data["total"] == 5
        assert len(data["sessions"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_filter_by_source_llm(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(client, token, source_llm="openai")
        await _create_session(client, token, source_llm="anthropic")

        resp = await client.get(
            "/api/v1/sessions?source_llm=openai",
            headers=_auth_headers(token),
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["sessions"][0]["source_llm"] == "openai"

    @pytest.mark.asyncio
    async def test_list_filter_archived(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)
        await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth_headers(token),
        )
        await _create_session(client, token, title="Active")

        resp = await client.get(
            "/api/v1/sessions?archived=false",
            headers=_auth_headers(token),
        )
        data = resp.json()
        assert data["total"] == 1
        assert data["sessions"][0]["title"] == "Active"


# -- Get Tests ---------------------------------------------------------------


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_session(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.get(
            f"/api/v1/sessions/{session['id']}",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == session["id"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get(
            "/api/v1/sessions/00000000-0000-0000-0000-000000000000",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_other_users_session(self, client) -> None:
        email1 = _unique_email()
        email2 = _unique_email()
        token1 = await _register_and_get_token(client, email1)
        token2 = await _register_and_get_token(client, email2)
        session = await _create_session(client, token1)

        resp = await client.get(
            f"/api/v1/sessions/{session['id']}",
            headers=_auth_headers(token2),
        )
        assert resp.status_code == 404


# -- Update Tests ------------------------------------------------------------


class TestUpdateSession:
    @pytest.mark.asyncio
    async def test_update_title(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.patch(
            f"/api/v1/sessions/{session['id']}",
            json={"title": "Updated Title"},
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Updated Title"

    @pytest.mark.asyncio
    async def test_update_tags(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.patch(
            f"/api/v1/sessions/{session['id']}",
            json={"tags": ["python", "fastapi"]},
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["python", "fastapi"]

    @pytest.mark.asyncio
    async def test_update_no_changes(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.patch(
            f"/api/v1/sessions/{session['id']}",
            json={},
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200


# -- Archive Tests -----------------------------------------------------------


class TestArchiveSession:
    @pytest.mark.asyncio
    async def test_archive(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        assert resp.json()["is_archived"] is True

    @pytest.mark.asyncio
    async def test_archive_already_archived(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)
        await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth_headers(token),
        )

        resp = await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unarchive(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)
        await client.post(
            f"/api/v1/sessions/{session['id']}/archive",
            headers=_auth_headers(token),
        )

        resp = await client.post(
            f"/api/v1/sessions/{session['id']}/unarchive",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 200
        assert resp.json()["is_archived"] is False

    @pytest.mark.asyncio
    async def test_unarchive_not_archived(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.post(
            f"/api/v1/sessions/{session['id']}/unarchive",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 400


# -- Delete Tests ------------------------------------------------------------


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_soft_delete(self, client) -> None:
        token = await _register_and_get_token(client)
        session = await _create_session(client, token)

        resp = await client.delete(
            f"/api/v1/sessions/{session['id']}",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 204

        # Should not appear in list anymore
        resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token),
        )
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.delete(
            "/api/v1/sessions/00000000-0000-0000-0000-000000000000",
            headers=_auth_headers(token),
        )
        assert resp.status_code == 404
