"""Cockpit orchestrator — parallel LLM dispatch and response collection.

Sends prompts to multiple LLMs simultaneously, collects responses,
updates shared context, and tracks costs.

In production, each model call goes through httpx async to the
respective API. For testing, mock adapters simulate responses.

Usage:
    orchestrator = CockpitOrchestrator()
    results = await orchestrator.dispatch(
        prompt="Explain async/await",
        models=["openai/gpt-4o", "anthropic/claude-sonnet-4"],
        system_context="You are a Python expert.",
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from app.core.cockpit.cost import CostEntry, CostTracker
from app.core.cockpit.roles import ModelRole, RoleManager
from app.core.cockpit.session import CockpitSession


@dataclass(frozen=True)
class ModelResponse:
    """Response from a single model.

    Attributes:
        model_id: Which model generated this response.
        content: The response text.
        prompt_tokens: Tokens used for the prompt.
        completion_tokens: Tokens used for the completion.
        latency_ms: Response time in milliseconds.
        error: Error message if the call failed.
        metadata: Additional response metadata.
    """

    model_id: str
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return bool(self.error)


@dataclass(frozen=True)
class DispatchResult:
    """Result of dispatching a prompt to multiple models.

    Attributes:
        prompt: The original prompt.
        responses: Map of model_id -> ModelResponse.
        total_latency_ms: Wall-clock time for all responses.
        total_cost_usd: Accumulated cost.
    """

    prompt: str
    responses: dict[str, ModelResponse]
    total_latency_ms: float
    total_cost_usd: float


# Type alias for model call functions
ModelCallFn = Callable[
    [str, str, str],  # model_id, prompt, system_context
    Awaitable[ModelResponse],
]


async def _default_model_call(
    model_id: str, prompt: str, system_context: str
) -> ModelResponse:
    """Default model call — routes to the real LLM client.

    Constructs an LLMMessage list from the prompt and system context,
    calls the unified LLM client, and wraps the result into a
    ModelResponse for the orchestrator.
    """
    from app.core.llm.client import LLMClient
    from app.core.llm.models import LLMError, LLMMessage

    client = LLMClient()
    messages: list[LLMMessage] = []

    if system_context:
        messages.append(LLMMessage(role="system", content=system_context))
    messages.append(LLMMessage(role="user", content=prompt))

    try:
        response = await client.call(model_id, messages)
        return ModelResponse(
            model_id=model_id,
            content=response.content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            metadata=response.metadata,
        )
    except LLMError as exc:
        return ModelResponse(
            model_id=model_id,
            error=str(exc),
        )


class CockpitOrchestrator:
    """Orchestrates parallel LLM dispatch for cockpit sessions.

    Features:
        - Parallel dispatch to 2-4 models simultaneously
        - Cost tracking per model and per session
        - Role-specific system prompts
        - Timeout handling for slow models
        - Error isolation (one model failing doesn't block others)
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None = None,
        role_manager: RoleManager | None = None,
        model_call_fn: ModelCallFn | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._cost_tracker = cost_tracker or CostTracker()
        self._role_manager = role_manager or RoleManager()
        self._model_call_fn = model_call_fn or _default_model_call
        self._timeout = timeout_seconds

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def role_manager(self) -> RoleManager:
        return self._role_manager

    async def dispatch(
        self,
        prompt: str,
        models: list[str],
        system_context: str = "",
        session_id: str = "",
        roles: dict[str, ModelRole] | None = None,
    ) -> DispatchResult:
        """Dispatch a prompt to multiple models in parallel.

        Args:
            prompt: The user's prompt.
            models: List of model identifiers to query.
            system_context: Shared context for all models.
            session_id: Optional session ID for cost tracking.
            roles: Optional role overrides per model.

        Returns:
            DispatchResult with all model responses.
        """
        if not models:
            return DispatchResult(
                prompt=prompt,
                responses={},
                total_latency_ms=0.0,
                total_cost_usd=0.0,
            )

        roles = roles or {}
        start = time.monotonic()

        # Build per-model system prompts
        tasks: list[asyncio.Task[ModelResponse]] = []
        for model_id in models:
            role = roles.get(model_id, ModelRole.GENERAL)
            role_prompt = self._role_manager.get_system_prompt(role, system_context)
            task = asyncio.create_task(
                self._call_model_safe(model_id, prompt, role_prompt)
            )
            tasks.append(task)

        # Await all in parallel
        responses_list = await asyncio.gather(*tasks, return_exceptions=False)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Build response map and track costs
        responses: dict[str, ModelResponse] = {}
        total_cost = 0.0

        for resp in responses_list:
            responses[resp.model_id] = resp
            if not resp.is_error and session_id:
                entry = self._cost_tracker.record(
                    session_id=session_id,
                    model_id=resp.model_id,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                )
                total_cost += entry.cost_usd

        return DispatchResult(
            prompt=prompt,
            responses=responses,
            total_latency_ms=round(elapsed_ms, 2),
            total_cost_usd=total_cost,
        )

    async def dispatch_single(
        self,
        prompt: str,
        model_id: str,
        system_context: str = "",
        role: ModelRole = ModelRole.GENERAL,
    ) -> ModelResponse:
        """Dispatch to a single model."""
        role_prompt = self._role_manager.get_system_prompt(role, system_context)
        return await self._call_model_safe(model_id, prompt, role_prompt)

    async def _call_model_safe(
        self,
        model_id: str,
        prompt: str,
        system_context: str,
    ) -> ModelResponse:
        """Call a model with timeout and error handling."""
        try:
            response = await asyncio.wait_for(
                self._model_call_fn(model_id, prompt, system_context),
                timeout=self._timeout,
            )
            return response
        except asyncio.TimeoutError:
            return ModelResponse(
                model_id=model_id,
                error=f"Timeout after {self._timeout}s",
                latency_ms=self._timeout * 1000,
            )
        except Exception as exc:
            return ModelResponse(
                model_id=model_id,
                error=str(exc),
            )
