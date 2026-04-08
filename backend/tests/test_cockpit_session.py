"""Tests for CockpitSession — immutable multi-model session state.

Tests cover:
    - Session creation
    - Add/remove models
    - Role assignment
    - Message history
    - Cost tracking
    - Status transitions
    - Shared context updates
    - Immutability guarantees
    - Serialization (to_dict)
    - Aggregate properties
"""

import pytest

from app.core.cockpit.session import (
    CockpitSession,
    CockpitStatus,
    ModelRole,
    ModelState,
)


# -- Creation tests ----------------------------------------------------------


class TestCockpitSessionCreate:
    def test_create_basic(self) -> None:
        session = CockpitSession.create(
            user_id="user1",
            models=["openai", "claude"],
        )
        assert session.user_id == "user1"
        assert session.status == CockpitStatus.ACTIVE
        assert len(session.models) == 2
        assert "openai" in session.models
        assert "claude" in session.models

    def test_create_with_roles(self) -> None:
        session = CockpitSession.create(
            user_id="user1",
            models=["openai", "claude"],
            roles={"openai": ModelRole.CODER, "claude": ModelRole.REVIEWER},
        )
        assert session.models["openai"].role == ModelRole.CODER
        assert session.models["claude"].role == ModelRole.REVIEWER

    def test_create_empty_models(self) -> None:
        session = CockpitSession.create(user_id="user1")
        assert len(session.models) == 0

    def test_create_with_context(self) -> None:
        session = CockpitSession.create(
            user_id="user1",
            models=["openai"],
            shared_context="You are a helpful assistant.",
        )
        assert session.shared_context == "You are a helpful assistant."

    def test_session_has_unique_id(self) -> None:
        a = CockpitSession.create(user_id="user1")
        b = CockpitSession.create(user_id="user1")
        assert a.session_id != b.session_id

    def test_timestamps_set(self) -> None:
        session = CockpitSession.create(user_id="user1")
        assert session.created_at > 0
        assert session.updated_at > 0


# -- Model management tests -------------------------------------------------


class TestCockpitSessionModels:
    def test_add_model(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        updated = session.add_model("claude", ModelRole.REVIEWER)
        assert "claude" in updated.models
        assert updated.models["claude"].role == ModelRole.REVIEWER
        # Original unchanged (immutable)
        assert "claude" not in session.models

    def test_remove_model(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["openai", "claude"]
        )
        updated = session.remove_model("claude")
        assert "claude" not in updated.models
        assert len(updated.models) == 1
        # Original unchanged
        assert "claude" in session.models

    def test_set_role(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        updated = session.set_role("openai", ModelRole.ANALYST)
        assert updated.models["openai"].role == ModelRole.ANALYST

    def test_set_role_nonexistent_model(self) -> None:
        session = CockpitSession.create(user_id="user1")
        updated = session.set_role("nonexistent", ModelRole.CODER)
        assert updated is session  # No change

    def test_model_ids_sorted(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["claude", "openai", "gemini"]
        )
        assert session.model_ids == ["claude", "gemini", "openai"]


# -- Message tests -----------------------------------------------------------


class TestCockpitSessionMessages:
    def test_add_message(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        updated = session.add_message("openai", "user", "Hello")
        assert len(updated.models["openai"].messages) == 1
        msg = updated.models["openai"].messages[0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_add_multiple_messages(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        s1 = session.add_message("openai", "user", "Hi")
        s2 = s1.add_message("openai", "assistant", "Hello!")
        assert len(s2.models["openai"].messages) == 2

    def test_add_message_nonexistent_model(self) -> None:
        session = CockpitSession.create(user_id="user1")
        updated = session.add_message("nope", "user", "Hi")
        assert updated is session  # No change

    def test_total_messages(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["openai", "claude"]
        )
        s1 = session.add_message("openai", "user", "Hi")
        s2 = s1.add_message("claude", "user", "Hi")
        s3 = s2.add_message("openai", "assistant", "Hello!")
        assert s3.total_messages == 3


# -- Cost tracking tests -----------------------------------------------------


class TestCockpitSessionCost:
    def test_update_cost(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        updated = session.update_cost(
            "openai", prompt_tokens=100, completion_tokens=50, cost_usd=0.003
        )
        m = updated.models["openai"]
        assert m.prompt_tokens == 100
        assert m.completion_tokens == 50
        assert m.cost_usd == pytest.approx(0.003)

    def test_cost_accumulates(self) -> None:
        session = CockpitSession.create(user_id="user1", models=["openai"])
        s1 = session.update_cost("openai", prompt_tokens=100, cost_usd=0.01)
        s2 = s1.update_cost("openai", prompt_tokens=200, cost_usd=0.02)
        assert s2.models["openai"].prompt_tokens == 300
        assert s2.models["openai"].cost_usd == pytest.approx(0.03)

    def test_total_cost(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["openai", "claude"]
        )
        s1 = session.update_cost("openai", cost_usd=0.01)
        s2 = s1.update_cost("claude", cost_usd=0.02)
        assert s2.total_cost_usd == pytest.approx(0.03)

    def test_total_tokens(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["openai", "claude"]
        )
        s1 = session.update_cost("openai", prompt_tokens=100, completion_tokens=50)
        s2 = s1.update_cost("claude", prompt_tokens=200, completion_tokens=80)
        assert s2.total_prompt_tokens == 300
        assert s2.total_completion_tokens == 130


# -- Status tests ------------------------------------------------------------


class TestCockpitSessionStatus:
    def test_set_status(self) -> None:
        session = CockpitSession.create(user_id="user1")
        ended = session.set_status(CockpitStatus.ENDED)
        assert ended.status == CockpitStatus.ENDED
        assert session.status == CockpitStatus.ACTIVE  # Original unchanged

    def test_pause_resume(self) -> None:
        session = CockpitSession.create(user_id="user1")
        paused = session.set_status(CockpitStatus.PAUSED)
        resumed = paused.set_status(CockpitStatus.ACTIVE)
        assert resumed.status == CockpitStatus.ACTIVE


# -- Context tests -----------------------------------------------------------


class TestCockpitSessionContext:
    def test_update_shared_context(self) -> None:
        session = CockpitSession.create(user_id="user1")
        updated = session.update_shared_context("New context info")
        assert updated.shared_context == "New context info"
        assert session.shared_context == ""  # Original unchanged


# -- Serialization tests -----------------------------------------------------


class TestCockpitSessionSerialization:
    def test_to_dict(self) -> None:
        session = CockpitSession.create(
            user_id="user1", models=["openai", "claude"]
        )
        d = session.to_dict()
        assert d["user_id"] == "user1"
        assert d["status"] == "active"
        assert "openai" in d["models"]
        assert "claude" in d["models"]
        assert d["total_cost_usd"] == 0.0
        assert d["total_messages"] == 0

    def test_to_dict_with_data(self) -> None:
        session = CockpitSession.create(
            user_id="user1",
            models=["openai"],
            roles={"openai": ModelRole.CODER},
        )
        s1 = session.add_message("openai", "user", "Hi")
        s2 = s1.update_cost("openai", prompt_tokens=100, cost_usd=0.005)
        d = s2.to_dict()
        assert d["models"]["openai"]["role"] == "coder"
        assert d["models"]["openai"]["message_count"] == 1
        assert d["models"]["openai"]["prompt_tokens"] == 100
        assert d["total_cost_usd"] == pytest.approx(0.005)
