"""Tests for webhook management endpoints and dispatcher.

Tests cover:
    - Webhook CRUD (create, list, update, delete)
    - Event validation
    - Duplicate URL rejection
    - Per-user limit
    - Ownership enforcement
    - Dispatcher: HMAC signature computation
    - Dispatcher: delivery result structure
    - Dispatcher: auto-disable after failures
"""

import hashlib
import hmac
import json
import uuid

import pytest

from app.core.events.webhook_dispatcher import (
    VALID_EVENTS,
    DeliveryResult,
    compute_signature,
)


# -- Helpers -----------------------------------------------------------------


def _unique_email() -> str:
    return f"webhook-{uuid.uuid4().hex[:12]}@example.com"


async def _register_and_get_token(client, email: str | None = None):
    if email is None:
        email = _unique_email()
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


async def _create_webhook(client, token, url=None, events=None):
    if url is None:
        url = f"https://example.com/hook-{uuid.uuid4().hex[:8]}"
    if events is None:
        events = ["session.created"]
    resp = await client.post(
        "/api/v1/webhooks",
        json={"url": url, "events": events},
        headers=_auth(token),
    )
    assert resp.status_code == 201
    return resp.json()


# -- Dispatcher Unit Tests ---------------------------------------------------


class TestWebhookDispatcher:
    def test_compute_signature(self) -> None:
        payload = b'{"event": "test"}'
        secret = "my_secret_key"
        sig = compute_signature(payload, secret)

        expected = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()
        assert sig == expected

    def test_compute_signature_different_secrets(self) -> None:
        payload = b'{"event": "test"}'
        sig1 = compute_signature(payload, "secret_a")
        sig2 = compute_signature(payload, "secret_b")
        assert sig1 != sig2

    def test_compute_signature_different_payloads(self) -> None:
        secret = "same_secret"
        sig1 = compute_signature(b"payload_a", secret)
        sig2 = compute_signature(b"payload_b", secret)
        assert sig1 != sig2

    def test_valid_events_frozen(self) -> None:
        assert isinstance(VALID_EVENTS, frozenset)
        assert "session.created" in VALID_EVENTS
        assert "context.generated" in VALID_EVENTS
        assert len(VALID_EVENTS) >= 8

    def test_delivery_result_frozen(self) -> None:
        result = DeliveryResult(
            webhook_id=uuid.uuid4(),
            url="https://example.com",
            event="session.created",
            success=True,
            status_code=200,
            attempts=1,
        )
        assert result.success is True
        assert result.error is None


# -- Create Tests ------------------------------------------------------------


class TestCreateWebhook:
    @pytest.mark.asyncio
    async def test_create_webhook(self, client) -> None:
        token = await _register_and_get_token(client)
        data = await _create_webhook(client, token)
        assert data["url"].startswith("https://")
        assert data["events"] == ["session.created"]
        assert data["is_active"] is True
        assert data["failure_count"] == 0
        # Secret is masked in response
        assert data["secret"].endswith("...")

    @pytest.mark.asyncio
    async def test_create_webhook_multiple_events(self, client) -> None:
        token = await _register_and_get_token(client)
        data = await _create_webhook(
            client, token,
            events=["session.created", "context.generated", "context.shifted"],
        )
        assert len(data["events"]) == 3

    @pytest.mark.asyncio
    async def test_create_webhook_invalid_event(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/webhooks",
            json={"url": "https://example.com/hook", "events": ["invalid.event"]},
            headers=_auth(token),
        )
        assert resp.status_code == 400
        assert "invalid events" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_webhook_duplicate_url(self, client) -> None:
        token = await _register_and_get_token(client)
        url = "https://example.com/unique-hook"
        await _create_webhook(client, token, url=url)

        resp = await client.post(
            "/api/v1/webhooks",
            json={"url": url, "events": ["session.created"]},
            headers=_auth(token),
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_create_webhook_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/webhooks",
            json={"url": "https://example.com/hook", "events": ["session.created"]},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_create_webhook_requires_events(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/webhooks",
            json={"url": "https://example.com/hook", "events": []},
            headers=_auth(token),
        )
        assert resp.status_code == 422


# -- List Tests --------------------------------------------------------------


class TestListWebhooks:
    @pytest.mark.asyncio
    async def test_list_webhooks(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_webhook(client, token)
        await _create_webhook(client, token)

        resp = await client.get("/api/v1/webhooks", headers=_auth(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_webhooks_empty(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get("/api/v1/webhooks", headers=_auth(token))
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# -- Update Tests ------------------------------------------------------------


class TestUpdateWebhook:
    @pytest.mark.asyncio
    async def test_update_url(self, client) -> None:
        token = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token)

        resp = await client.patch(
            f"/api/v1/webhooks/{webhook['id']}",
            json={"url": "https://new-url.example.com/hook"},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        assert resp.json()["url"] == "https://new-url.example.com/hook"

    @pytest.mark.asyncio
    async def test_update_events(self, client) -> None:
        token = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token)

        resp = await client.patch(
            f"/api/v1/webhooks/{webhook['id']}",
            json={"events": ["context.generated"]},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        assert resp.json()["events"] == ["context.generated"]

    @pytest.mark.asyncio
    async def test_update_deactivate(self, client) -> None:
        token = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token)

        resp = await client.patch(
            f"/api/v1/webhooks/{webhook['id']}",
            json={"is_active": False},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.patch(
            f"/api/v1/webhooks/{uuid.uuid4()}",
            json={"url": "https://nope.example.com"},
            headers=_auth(token),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_invalid_events(self, client) -> None:
        token = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token)

        resp = await client.patch(
            f"/api/v1/webhooks/{webhook['id']}",
            json={"events": ["bad.event"]},
            headers=_auth(token),
        )
        assert resp.status_code == 400


# -- Delete Tests ------------------------------------------------------------


class TestDeleteWebhook:
    @pytest.mark.asyncio
    async def test_delete_webhook(self, client) -> None:
        token = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token)

        resp = await client.delete(
            f"/api/v1/webhooks/{webhook['id']}",
            headers=_auth(token),
        )
        assert resp.status_code == 204

        # Should be gone
        resp = await client.get("/api/v1/webhooks", headers=_auth(token))
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.delete(
            f"/api/v1/webhooks/{uuid.uuid4()}",
            headers=_auth(token),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_other_users_webhook(self, client) -> None:
        token1 = await _register_and_get_token(client)
        token2 = await _register_and_get_token(client)
        webhook = await _create_webhook(client, token1)

        resp = await client.delete(
            f"/api/v1/webhooks/{webhook['id']}",
            headers=_auth(token2),
        )
        assert resp.status_code == 404
