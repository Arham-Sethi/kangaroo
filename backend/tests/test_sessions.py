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


# -- Capture Tests (Browser Extension) ----------------------------------------


def _capture_payload(**overrides) -> dict:
    """Build a valid capture request payload."""
    body = {
        "platform": "chatgpt",
        "model": "gpt-4o",
        "title": "Test Conversation",
        "url": "https://chatgpt.com/c/abc123",
        "messages": [
            {"role": "user", "content": "Hello, how are you?", "model": ""},
            {
                "role": "assistant",
                "content": "I'm doing well! How can I help you today?",
                "model": "gpt-4o",
            },
        ],
        "captured_at": "2026-04-07T12:00:00Z",
    }
    body.update(overrides)
    return body


class TestCaptureConversation:
    """Tests for POST /api/v1/sessions/capture — browser extension capture endpoint."""

    @pytest.mark.asyncio
    async def test_capture_basic(self, client) -> None:
        """Capture a simple 2-message conversation."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["message_count"] == 2
        assert data["title"] == "Test Conversation"
        assert data["source_llm"] == "openai"
        assert data["source_model"] == "gpt-4o"
        assert "session_id" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_capture_chatgpt_maps_to_openai(self, client) -> None:
        """Platform 'chatgpt' maps to source_llm 'openai'."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(platform="chatgpt"),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        assert resp.json()["source_llm"] == "openai"

    @pytest.mark.asyncio
    async def test_capture_claude_maps_to_anthropic(self, client) -> None:
        """Platform 'claude' maps to source_llm 'anthropic'."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(platform="claude", model="claude-sonnet-4"),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["source_llm"] == "anthropic"
        assert data["source_model"] == "claude-sonnet-4"

    @pytest.mark.asyncio
    async def test_capture_gemini_maps_to_google(self, client) -> None:
        """Platform 'gemini' maps to source_llm 'google'."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(platform="gemini", model="gemini-2.0-flash"),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        assert resp.json()["source_llm"] == "google"

    @pytest.mark.asyncio
    async def test_capture_creates_session_in_list(self, client) -> None:
        """Captured conversation appears in session list."""
        token = await _register_and_get_token(client)
        capture_resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(title="My Chat"),
            headers=_auth_headers(token),
        )
        assert capture_resp.status_code == 201
        session_id = capture_resp.json()["session_id"]

        list_resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token),
        )
        assert list_resp.status_code == 200
        sessions = list_resp.json()["sessions"]
        assert any(s["id"] == session_id for s in sessions)

    @pytest.mark.asyncio
    async def test_capture_session_has_auto_capture_tag(self, client) -> None:
        """Captured session is tagged with 'auto-capture' and platform name."""
        token = await _register_and_get_token(client)
        capture_resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(platform="claude"),
            headers=_auth_headers(token),
        )
        session_id = capture_resp.json()["session_id"]

        get_resp = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers=_auth_headers(token),
        )
        assert get_resp.status_code == 200
        tags = get_resp.json()["tags"]
        assert "auto-capture" in tags
        assert "claude" in tags

    @pytest.mark.asyncio
    async def test_capture_many_messages(self, client) -> None:
        """Capture a long conversation with many messages."""
        token = await _register_and_get_token(client)
        messages = []
        for i in range(50):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i} " * 20, "model": "gpt-4o"})

        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(messages=messages),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        assert resp.json()["message_count"] == 50

    @pytest.mark.asyncio
    async def test_capture_estimates_tokens(self, client) -> None:
        """Token count is estimated from message content length (~4 chars/token)."""
        token = await _register_and_get_token(client)
        # Create a message with known content length
        content = "a" * 400  # ~100 tokens
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(
                messages=[{"role": "user", "content": content, "model": ""}]
            ),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]

        # Check the session's total_tokens is approximately correct
        get_resp = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers=_auth_headers(token),
        )
        total_tokens = get_resp.json()["total_tokens"]
        assert total_tokens == 100  # 400 chars / 4 chars per token

    @pytest.mark.asyncio
    async def test_capture_requires_auth(self, client) -> None:
        """Capture endpoint requires authentication."""
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(),
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_capture_invalid_platform(self, client) -> None:
        """Invalid platform is rejected by validation."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(platform="invalid_platform"),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_capture_empty_messages(self, client) -> None:
        """Empty messages list is rejected."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(messages=[]),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_capture_invalid_role(self, client) -> None:
        """Message with invalid role is rejected."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(
                messages=[{"role": "invalid", "content": "hello", "model": ""}]
            ),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_capture_empty_content_rejected(self, client) -> None:
        """Message with empty content is rejected (min_length=1)."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(
                messages=[{"role": "user", "content": "", "model": ""}]
            ),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_capture_missing_url(self, client) -> None:
        """Missing URL field is rejected."""
        token = await _register_and_get_token(client)
        payload = _capture_payload()
        del payload["url"]
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=payload,
            headers=_auth_headers(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_capture_default_title(self, client) -> None:
        """When title is empty, falls back to default."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(title=""),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201
        assert resp.json()["title"] == "Captured Conversation"

    @pytest.mark.asyncio
    async def test_capture_iso_timestamp_parsed(self, client) -> None:
        """ISO-8601 captured_at timestamp is stored correctly."""
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(captured_at="2026-01-15T09:30:00+05:00"),
            headers=_auth_headers(token),
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_capture_multiple_conversations(self, client) -> None:
        """Multiple captures create separate sessions."""
        token = await _register_and_get_token(client)
        ids = set()
        for i in range(3):
            resp = await client.post(
                "/api/v1/sessions/capture",
                json=_capture_payload(title=f"Conversation {i}"),
                headers=_auth_headers(token),
            )
            assert resp.status_code == 201
            ids.add(resp.json()["session_id"])

        assert len(ids) == 3  # All unique session IDs

        list_resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token),
        )
        assert list_resp.json()["total"] == 3

    @pytest.mark.asyncio
    async def test_capture_isolation_between_users(self, client) -> None:
        """User A's captures are not visible to User B."""
        token_a = await _register_and_get_token(client)
        token_b = await _register_and_get_token(client)

        # User A captures
        resp = await client.post(
            "/api/v1/sessions/capture",
            json=_capture_payload(title="User A's chat"),
            headers=_auth_headers(token_a),
        )
        assert resp.status_code == 201
        session_id = resp.json()["session_id"]

        # User B can't see it
        resp = await client.get(
            f"/api/v1/sessions/{session_id}",
            headers=_auth_headers(token_b),
        )
        assert resp.status_code == 404

        # User B's list is empty
        resp = await client.get(
            "/api/v1/sessions",
            headers=_auth_headers(token_b),
        )
        assert resp.json()["total"] == 0
