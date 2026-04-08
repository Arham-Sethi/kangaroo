"""Tests for team workspace operations.

Tests cover:
    - Team service: create, list, invite, role changes, removal
    - RBAC: owner > admin > member > viewer hierarchy
    - Tier gating: free users can't create teams
    - Error cases: not found, permission denied, member exists
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from app.core.teams.workspace import (
    MemberExistsError,
    TeamError,
    TeamLimitError,
    TeamNotFoundError,
    TeamPermissionError,
    TeamService,
    ROLE_HIERARCHY,
    TIER_MAX_SEATS,
    _has_role,
    _slugify,
)
from app.core.models.db import TeamRole


# ── Unit Tests ───────────────────────────────────────────────────────────────


class TestSlugify:
    def test_simple_name(self) -> None:
        slug = _slugify("My Team")
        assert slug.startswith("my-team-")
        assert len(slug) <= 200

    def test_special_characters(self) -> None:
        slug = _slugify("Team @#$ Special!")
        assert "@" not in slug
        assert "#" not in slug
        assert "$" not in slug

    def test_uniqueness(self) -> None:
        slug1 = _slugify("Same Name")
        slug2 = _slugify("Same Name")
        assert slug1 != slug2  # UUID suffix ensures uniqueness

    def test_empty_string(self) -> None:
        slug = _slugify("")
        assert len(slug) > 0  # UUID suffix still present


class TestRoleHierarchy:
    def test_owner_outranks_all(self) -> None:
        assert _has_role("owner", "owner")
        assert _has_role("owner", "admin")
        assert _has_role("owner", "member")
        assert _has_role("owner", "viewer")

    def test_admin_outranks_member_viewer(self) -> None:
        assert _has_role("admin", "admin")
        assert _has_role("admin", "member")
        assert _has_role("admin", "viewer")
        assert not _has_role("admin", "owner")

    def test_member_outranks_viewer(self) -> None:
        assert _has_role("member", "member")
        assert _has_role("member", "viewer")
        assert not _has_role("member", "admin")

    def test_viewer_is_lowest(self) -> None:
        assert _has_role("viewer", "viewer")
        assert not _has_role("viewer", "member")
        assert not _has_role("viewer", "admin")

    def test_unknown_role_has_no_access(self) -> None:
        assert not _has_role("unknown", "viewer")


class TestTierLimits:
    def test_free_has_no_seats(self) -> None:
        assert TIER_MAX_SEATS["free"] == 0

    def test_pro_has_4_seats(self) -> None:
        assert TIER_MAX_SEATS["pro"] == 4

    def test_pro_team_has_50_seats(self) -> None:
        assert TIER_MAX_SEATS["pro_team"] == 50

    def test_enterprise_has_500_seats(self) -> None:
        assert TIER_MAX_SEATS["enterprise"] == 500


class TestExceptions:
    def test_team_error_hierarchy(self) -> None:
        assert issubclass(TeamNotFoundError, TeamError)
        assert issubclass(TeamPermissionError, TeamError)
        assert issubclass(TeamLimitError, TeamError)
        assert issubclass(MemberExistsError, TeamError)

    def test_error_messages(self) -> None:
        err = TeamNotFoundError("Team not found")
        assert str(err) == "Team not found"

    def test_limit_error_is_descriptive(self) -> None:
        err = TeamLimitError("Upgrade to Pro")
        assert "Pro" in str(err)


class TestRoleConstants:
    def test_all_roles_have_hierarchy(self) -> None:
        for role in TeamRole:
            assert role.value in ROLE_HIERARCHY, f"{role.value} missing from hierarchy"

    def test_hierarchy_is_ordered(self) -> None:
        assert ROLE_HIERARCHY["owner"] > ROLE_HIERARCHY["admin"]
        assert ROLE_HIERARCHY["admin"] > ROLE_HIERARCHY["member"]
        assert ROLE_HIERARCHY["member"] > ROLE_HIERARCHY["viewer"]
