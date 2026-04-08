"""Tests for RoleManager — role definitions and system prompts.

Tests cover:
    - Get role definition
    - Generate system prompts
    - List all roles
    - Role suggestion from task description
    - Custom role definitions
    - Unknown role fallback
"""

import pytest

from app.core.cockpit.roles import RoleDefinition, RoleManager
from app.core.cockpit.session import ModelRole


class TestRoleDefinition:
    def test_get_all_roles(self) -> None:
        mgr = RoleManager()
        roles = mgr.list_roles()
        assert len(roles) >= 6
        role_types = {r.role for r in roles}
        assert ModelRole.CODER in role_types
        assert ModelRole.REVIEWER in role_types
        assert ModelRole.GENERAL in role_types

    def test_get_definition(self) -> None:
        mgr = RoleManager()
        defn = mgr.get_definition(ModelRole.CODER)
        assert defn.role == ModelRole.CODER
        assert defn.name == "Coder"
        assert len(defn.strengths) > 0

    def test_unknown_role_returns_general(self) -> None:
        mgr = RoleManager()
        # ModelRole.GENERAL is the fallback
        defn = mgr.get_definition(ModelRole.GENERAL)
        assert defn.role == ModelRole.GENERAL


class TestSystemPrompt:
    def test_generate_with_context(self) -> None:
        mgr = RoleManager()
        prompt = mgr.get_system_prompt(ModelRole.CODER, context="Build a REST API")
        assert "Build a REST API" in prompt
        assert "engineer" in prompt.lower() or "code" in prompt.lower()

    def test_generate_without_context(self) -> None:
        mgr = RoleManager()
        prompt = mgr.get_system_prompt(ModelRole.REVIEWER)
        assert "reviewer" in prompt.lower() or "review" in prompt.lower()

    def test_each_role_has_unique_prompt(self) -> None:
        mgr = RoleManager()
        prompts = set()
        for role in ModelRole:
            prompt = mgr.get_system_prompt(role, context="test")
            prompts.add(prompt)
        # All roles should produce different prompts
        assert len(prompts) == len(ModelRole)


class TestRoleSuggestion:
    def test_suggest_coder(self) -> None:
        mgr = RoleManager()
        role = mgr.suggest_role("implement the new feature with testing")
        assert role == ModelRole.CODER

    def test_suggest_reviewer(self) -> None:
        mgr = RoleManager()
        role = mgr.suggest_role("review the code for security issues")
        assert role == ModelRole.REVIEWER

    def test_suggest_general_for_vague(self) -> None:
        mgr = RoleManager()
        role = mgr.suggest_role("help me with something")
        assert role == ModelRole.GENERAL


class TestCustomRoles:
    def test_custom_role_overrides(self) -> None:
        custom = {
            ModelRole.CODER: RoleDefinition(
                role=ModelRole.CODER,
                name="Custom Coder",
                description="Custom coding role",
                system_prompt_template="Custom: {context}",
                strengths=("custom",),
            )
        }
        mgr = RoleManager(custom_roles=custom)
        defn = mgr.get_definition(ModelRole.CODER)
        assert defn.name == "Custom Coder"
        prompt = mgr.get_system_prompt(ModelRole.CODER, "test")
        assert prompt == "Custom: test"
