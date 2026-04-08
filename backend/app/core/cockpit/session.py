"""Cockpit session — multi-model session state management.

A CockpitSession tracks:
    - Which models are active and their assigned roles
    - Shared context (UCS) across all models
    - Message history for each model
    - Cost accumulation per model
    - Session lifecycle (created, active, ended)

Usage:
    session = CockpitSession.create(user_id="abc", models=["openai", "claude"])
    session = session.add_message("openai", role="assistant", content="Hello")
    session = session.update_cost("openai", prompt_tokens=100, completion_tokens=50)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class CockpitStatus(str, Enum):
    """Lifecycle status of a cockpit session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class ModelRole(str, Enum):
    """Roles that can be assigned to models in a cockpit."""

    CODER = "coder"
    REVIEWER = "reviewer"
    EXPLAINER = "explainer"
    ANALYST = "analyst"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass(frozen=True)
class ModelState:
    """State of a single model in the cockpit.

    Attributes:
        model_id: Identifier (e.g., "openai", "claude", "gemini").
        role: Assigned role in this session.
        is_active: Whether this model is currently responding.
        messages: Message history for this model.
        prompt_tokens: Total prompt tokens used.
        completion_tokens: Total completion tokens used.
        cost_usd: Accumulated cost in USD.
    """

    model_id: str
    role: ModelRole = ModelRole.GENERAL
    is_active: bool = True
    messages: tuple[dict[str, Any], ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class CockpitSession:
    """Immutable cockpit session state.

    All mutations return a new CockpitSession instance.

    Attributes:
        session_id: Unique session identifier.
        user_id: Owner of this session.
        status: Current lifecycle status.
        models: Map of model_id -> ModelState.
        shared_context: Shared system context injected into all models.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
        metadata: Additional session metadata.
    """

    session_id: str
    user_id: str
    status: CockpitStatus = CockpitStatus.CREATED
    models: dict[str, ModelState] = field(default_factory=dict)
    shared_context: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        user_id: str,
        models: list[str] | None = None,
        roles: dict[str, ModelRole] | None = None,
        shared_context: str = "",
    ) -> CockpitSession:
        """Create a new cockpit session.

        Args:
            user_id: Owner of the session.
            models: List of model identifiers to include.
            roles: Optional role assignments per model.
            shared_context: Shared system context for all models.

        Returns:
            A new CockpitSession instance.
        """
        now = time.time()
        model_states: dict[str, ModelState] = {}
        roles = roles or {}

        for model_id in (models or []):
            role = roles.get(model_id, ModelRole.GENERAL)
            model_states[model_id] = ModelState(
                model_id=model_id,
                role=role,
            )

        return CockpitSession(
            session_id=uuid4().hex[:16],
            user_id=user_id,
            status=CockpitStatus.ACTIVE,
            models=model_states,
            shared_context=shared_context,
            created_at=now,
            updated_at=now,
        )

    def add_model(
        self,
        model_id: str,
        role: ModelRole = ModelRole.GENERAL,
    ) -> CockpitSession:
        """Add a model to the session."""
        new_models = dict(self.models)
        new_models[model_id] = ModelState(model_id=model_id, role=role)
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=new_models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def remove_model(self, model_id: str) -> CockpitSession:
        """Remove a model from the session."""
        new_models = {k: v for k, v in self.models.items() if k != model_id}
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=new_models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def set_role(self, model_id: str, role: ModelRole) -> CockpitSession:
        """Change a model's role."""
        if model_id not in self.models:
            return self

        old = self.models[model_id]
        new_state = ModelState(
            model_id=old.model_id,
            role=role,
            is_active=old.is_active,
            messages=old.messages,
            prompt_tokens=old.prompt_tokens,
            completion_tokens=old.completion_tokens,
            cost_usd=old.cost_usd,
        )
        new_models = dict(self.models)
        new_models[model_id] = new_state
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=new_models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def add_message(
        self,
        model_id: str,
        role: str,
        content: str,
    ) -> CockpitSession:
        """Add a message to a model's history."""
        if model_id not in self.models:
            return self

        old = self.models[model_id]
        msg = {"role": role, "content": content, "timestamp": time.time()}
        new_state = ModelState(
            model_id=old.model_id,
            role=old.role,
            is_active=old.is_active,
            messages=old.messages + (msg,),
            prompt_tokens=old.prompt_tokens,
            completion_tokens=old.completion_tokens,
            cost_usd=old.cost_usd,
        )
        new_models = dict(self.models)
        new_models[model_id] = new_state
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=new_models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def update_cost(
        self,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> CockpitSession:
        """Accumulate token/cost usage for a model."""
        if model_id not in self.models:
            return self

        old = self.models[model_id]
        new_state = ModelState(
            model_id=old.model_id,
            role=old.role,
            is_active=old.is_active,
            messages=old.messages,
            prompt_tokens=old.prompt_tokens + prompt_tokens,
            completion_tokens=old.completion_tokens + completion_tokens,
            cost_usd=old.cost_usd + cost_usd,
        )
        new_models = dict(self.models)
        new_models[model_id] = new_state
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=new_models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def set_status(self, status: CockpitStatus) -> CockpitSession:
        """Change session status."""
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=status,
            models=self.models,
            shared_context=self.shared_context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    def update_shared_context(self, context: str) -> CockpitSession:
        """Update the shared context for all models."""
        return CockpitSession(
            session_id=self.session_id,
            user_id=self.user_id,
            status=self.status,
            models=self.models,
            shared_context=context,
            created_at=self.created_at,
            updated_at=time.time(),
            metadata=self.metadata,
        )

    @property
    def model_ids(self) -> list[str]:
        return sorted(self.models.keys())

    @property
    def total_prompt_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.models.values())

    @property
    def total_completion_tokens(self) -> int:
        return sum(m.completion_tokens for m in self.models.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(m.cost_usd for m in self.models.values())

    @property
    def total_messages(self) -> int:
        return sum(len(m.messages) for m in self.models.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dict for WebSocket/API responses."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "models": {
                mid: {
                    "model_id": ms.model_id,
                    "role": ms.role.value,
                    "is_active": ms.is_active,
                    "message_count": len(ms.messages),
                    "prompt_tokens": ms.prompt_tokens,
                    "completion_tokens": ms.completion_tokens,
                    "cost_usd": round(ms.cost_usd, 6),
                }
                for mid, ms in self.models.items()
            },
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_messages": self.total_messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
