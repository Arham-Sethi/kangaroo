"""Role management — assign specialized roles to models in a cockpit.

Each role provides a system prompt that shapes how the model behaves.
This enables diverse perspectives: one model codes, another reviews,
a third explains — all working on the same problem.

Usage:
    manager = RoleManager()
    prompt = manager.get_system_prompt(ModelRole.CODER, context="Build a REST API")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.cockpit.session import ModelRole


@dataclass(frozen=True)
class RoleDefinition:
    """Definition of a cockpit role.

    Attributes:
        role: The role enum value.
        name: Human-readable role name.
        description: What this role does.
        system_prompt_template: System prompt template with {context} placeholder.
        strengths: What this role is good at.
    """

    role: ModelRole
    name: str
    description: str
    system_prompt_template: str
    strengths: tuple[str, ...]


# Built-in role definitions
_ROLE_DEFINITIONS: dict[ModelRole, RoleDefinition] = {
    ModelRole.CODER: RoleDefinition(
        role=ModelRole.CODER,
        name="Coder",
        description="Writes clean, production-ready code.",
        system_prompt_template=(
            "You are an expert software engineer. Write clean, well-tested, "
            "production-ready code. Follow best practices and design patterns.\n\n"
            "Context:\n{context}"
        ),
        strengths=("implementation", "debugging", "optimization", "testing"),
    ),
    ModelRole.REVIEWER: RoleDefinition(
        role=ModelRole.REVIEWER,
        name="Code Reviewer",
        description="Reviews code for quality, security, and best practices.",
        system_prompt_template=(
            "You are a senior code reviewer. Analyze code for bugs, security "
            "vulnerabilities, performance issues, and design problems. Be thorough "
            "but constructive.\n\nContext:\n{context}"
        ),
        strengths=("code review", "security", "best practices", "architecture"),
    ),
    ModelRole.EXPLAINER: RoleDefinition(
        role=ModelRole.EXPLAINER,
        name="Explainer",
        description="Explains concepts clearly and concisely.",
        system_prompt_template=(
            "You are a technical educator. Explain concepts clearly with examples. "
            "Adapt your explanation to the user's expertise level. Use analogies "
            "when helpful.\n\nContext:\n{context}"
        ),
        strengths=("documentation", "teaching", "simplification", "examples"),
    ),
    ModelRole.ANALYST: RoleDefinition(
        role=ModelRole.ANALYST,
        name="Analyst",
        description="Analyzes data, requirements, and trade-offs.",
        system_prompt_template=(
            "You are a systems analyst. Analyze requirements, identify trade-offs, "
            "evaluate alternatives, and provide data-driven recommendations.\n\n"
            "Context:\n{context}"
        ),
        strengths=("analysis", "trade-offs", "requirements", "research"),
    ),
    ModelRole.CREATIVE: RoleDefinition(
        role=ModelRole.CREATIVE,
        name="Creative",
        description="Generates creative solutions and ideas.",
        system_prompt_template=(
            "You are a creative problem solver. Think outside the box, propose "
            "innovative approaches, and explore unconventional solutions.\n\n"
            "Context:\n{context}"
        ),
        strengths=("brainstorming", "innovation", "ideation", "design"),
    ),
    ModelRole.GENERAL: RoleDefinition(
        role=ModelRole.GENERAL,
        name="General Assistant",
        description="Versatile assistant for any task.",
        system_prompt_template=(
            "You are a helpful AI assistant. Respond accurately and concisely.\n\n"
            "Context:\n{context}"
        ),
        strengths=("versatility", "general knowledge", "conversation"),
    ),
}


class RoleManager:
    """Manages role definitions and generates role-specific system prompts."""

    def __init__(
        self,
        custom_roles: dict[ModelRole, RoleDefinition] | None = None,
    ) -> None:
        self._roles = dict(_ROLE_DEFINITIONS)
        if custom_roles:
            self._roles.update(custom_roles)

    def get_definition(self, role: ModelRole) -> RoleDefinition:
        """Get the role definition."""
        return self._roles.get(role, _ROLE_DEFINITIONS[ModelRole.GENERAL])

    def get_system_prompt(self, role: ModelRole, context: str = "") -> str:
        """Generate a system prompt for a role with the given context."""
        definition = self.get_definition(role)
        return definition.system_prompt_template.format(context=context)

    def list_roles(self) -> list[RoleDefinition]:
        """List all available roles."""
        return list(self._roles.values())

    def suggest_role(self, task_description: str) -> ModelRole:
        """Suggest the best role for a task based on keywords.

        Simple keyword matching — production would use embeddings.
        """
        task_lower = task_description.lower()

        # Check each role's strengths for keyword overlap
        best_role = ModelRole.GENERAL
        best_score = 0

        for role, definition in self._roles.items():
            score = sum(
                1 for strength in definition.strengths
                if strength in task_lower
            )
            if score > best_score:
                best_score = score
                best_role = role

        return best_role
