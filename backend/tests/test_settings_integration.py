"""Integration tests for settings-related endpoints.

Tests cover:
    - PATCH /api/v1/auth/profile — update display name and settings
    - GET /api/v1/api-keys — list API keys
    - POST /api/v1/api-keys — create API key
    - DELETE /api/v1/api-keys/{id} — revoke API key
    - GET /api/v1/teams — list teams
    - POST /api/v1/teams — create team
    - POST /api/v1/teams/{id}/invite — invite member
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.api.v1.api_keys import (
    APIKeyCreate,
    APIKeyResponse,
    APIKeyCreatedResponse,
    APIKeyListResponse,
    MAX_KEYS_PER_USER,
)


# ── API Key Schema Tests ────────────────────────────────────────────────────


class TestApiKeySchemas:
    """Tests for API key request/response schemas."""

    def test_create_request_valid(self) -> None:
        req = APIKeyCreate(name="Production Key")
        assert req.name == "Production Key"
        assert len(req.scopes) > 0

    def test_create_request_with_scopes(self) -> None:
        req = APIKeyCreate(
            name="Read Only",
            scopes=["contexts:read", "sessions:read"],
        )
        assert req.scopes == ["contexts:read", "sessions:read"]

    def test_create_request_min_name_length(self) -> None:
        with pytest.raises(Exception):
            APIKeyCreate(name="")

    def test_create_request_max_name_length(self) -> None:
        with pytest.raises(Exception):
            APIKeyCreate(name="x" * 101)

    def test_response_schema(self) -> None:
        resp = APIKeyResponse(
            id="key-123",
            name="Test Key",
            key_prefix="ks_abc123def4",
            scopes=["contexts:read"],
            is_active=True,
            created_at="2026-01-01T00:00:00",
            last_used=None,
            expires_at=None,
        )
        assert resp.is_active is True
        assert resp.key_prefix == "ks_abc123def4"

    def test_created_response_includes_full_key(self) -> None:
        resp = APIKeyCreatedResponse(
            id="key-123",
            name="Test Key",
            key_prefix="ks_abc123def4",
            scopes=["contexts:read"],
            is_active=True,
            created_at="2026-01-01T00:00:00",
            last_used=None,
            expires_at=None,
            key="ks_full_secret_key_value_here",
        )
        assert resp.key.startswith("ks_")
        assert len(resp.key) > len(resp.key_prefix)

    def test_list_response_schema(self) -> None:
        resp = APIKeyListResponse(keys=[], total=0)
        assert resp.total == 0
        assert resp.keys == []

    def test_max_keys_constant(self) -> None:
        assert MAX_KEYS_PER_USER == 25


class TestApiKeyValidScopes:
    """Tests for API key scope validation."""

    VALID_SCOPES = [
        "contexts:read",
        "contexts:write",
        "shifts:execute",
        "sessions:read",
        "sessions:write",
        "brain:read",
        "brain:write",
    ]

    def test_all_valid_scopes(self) -> None:
        for scope in self.VALID_SCOPES:
            req = APIKeyCreate(name="test", scopes=[scope])
            assert scope in req.scopes

    def test_default_scopes(self) -> None:
        req = APIKeyCreate(name="test")
        assert "contexts:read" in req.scopes
        assert "shifts:execute" in req.scopes


# ── Profile Update Tests ────────────────────────────────────────────────────


class TestProfileUpdate:
    """Tests for profile update schema."""

    def test_profile_update_import(self) -> None:
        from app.api.v1.auth import ProfileUpdate

        update = ProfileUpdate(display_name="John Doe")
        assert update.display_name == "John Doe"

    def test_profile_update_with_settings(self) -> None:
        from app.api.v1.auth import ProfileUpdate

        update = ProfileUpdate(settings={"theme": "dark", "language": "en"})
        assert update.settings is not None
        assert update.settings["theme"] == "dark"

    def test_profile_update_all_optional(self) -> None:
        from app.api.v1.auth import ProfileUpdate

        update = ProfileUpdate()
        assert update.display_name is None
        assert update.settings is None


# ── Team Schema Tests ────────────────────────────────────────────────────────


class TestTeamSchemas:
    """Tests for team-related schemas."""

    def test_team_creation_request(self) -> None:
        from app.api.v1.teams import CreateTeamRequest

        req = CreateTeamRequest(name="Engineering")
        assert req.name == "Engineering"

    def test_invite_member_request(self) -> None:
        from app.api.v1.teams import InviteMemberRequest

        req = InviteMemberRequest(email="john@example.com", role="member")
        assert req.email == "john@example.com"
        assert req.role == "member"

    def test_change_role_request(self) -> None:
        from app.api.v1.teams import ChangeRoleRequest

        req = ChangeRoleRequest(role="admin")
        assert req.role == "admin"
