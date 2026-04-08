"""Tests for CockpitOrchestrator — parallel LLM dispatch.

Tests cover:
    - Parallel dispatch to multiple models
    - Single model dispatch
    - Cost tracking integration
    - Timeout handling
    - Error isolation
    - Role-based system prompts
    - Empty model list
"""

import asyncio

import pytest

from app.core.cockpit.cost import CostTracker
from app.core.cockpit.orchestrator import (
    CockpitOrchestrator,
    DispatchResult,
    ModelResponse,
)
from app.core.cockpit.roles import RoleManager
from app.core.cockpit.session import ModelRole


# -- Mock model calls --------------------------------------------------------


async def _mock_call(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    """Mock LLM call that returns a predictable response."""
    return ModelResponse(
        model_id=model_id,
        content=f"Response from {model_id}",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=10.0,
    )


async def _slow_call(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    """Mock LLM call that takes too long."""
    await asyncio.sleep(5)
    return ModelResponse(model_id=model_id, content="Late response")


async def _error_call(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    """Mock LLM call that raises an exception."""
    raise RuntimeError(f"API error for {model_id}")


async def _partial_error_call(
    model_id: str, prompt: str, system_context: str
) -> ModelResponse:
    """One model fails, others succeed."""
    if model_id == "bad_model":
        raise RuntimeError("Bad model failed")
    return ModelResponse(
        model_id=model_id,
        content=f"OK from {model_id}",
        prompt_tokens=50,
        completion_tokens=25,
    )


# -- Tests -------------------------------------------------------------------


class TestOrchestratorDispatch:
    @pytest.mark.asyncio
    async def test_parallel_dispatch(self) -> None:
        orch = CockpitOrchestrator(model_call_fn=_mock_call)
        result = await orch.dispatch(
            prompt="Hello",
            models=["openai/gpt-4o", "anthropic/claude-sonnet-4"],
        )
        assert isinstance(result, DispatchResult)
        assert len(result.responses) == 2
        assert "openai/gpt-4o" in result.responses
        assert "anthropic/claude-sonnet-4" in result.responses
        assert result.responses["openai/gpt-4o"].content == "Response from openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_empty_models(self) -> None:
        orch = CockpitOrchestrator(model_call_fn=_mock_call)
        result = await orch.dispatch(prompt="Hello", models=[])
        assert result.responses == {}
        assert result.total_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_single_dispatch(self) -> None:
        orch = CockpitOrchestrator(model_call_fn=_mock_call)
        resp = await orch.dispatch_single("Hello", "openai/gpt-4o")
        assert resp.model_id == "openai/gpt-4o"
        assert resp.content == "Response from openai/gpt-4o"
        assert not resp.is_error


class TestOrchestratorCost:
    @pytest.mark.asyncio
    async def test_cost_tracking(self) -> None:
        tracker = CostTracker()
        orch = CockpitOrchestrator(cost_tracker=tracker, model_call_fn=_mock_call)
        result = await orch.dispatch(
            prompt="Hello",
            models=["openai/gpt-4o"],
            session_id="sess1",
        )
        total = tracker.get_session_total("sess1")
        assert total > 0
        assert result.total_cost_usd > 0

    @pytest.mark.asyncio
    async def test_cost_not_tracked_without_session(self) -> None:
        tracker = CostTracker()
        orch = CockpitOrchestrator(cost_tracker=tracker, model_call_fn=_mock_call)
        await orch.dispatch(prompt="Hello", models=["openai/gpt-4o"])
        # No session_id means no recording
        assert tracker.get_session_total("") == 0.0


class TestOrchestratorTimeout:
    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        orch = CockpitOrchestrator(
            model_call_fn=_slow_call, timeout_seconds=0.1
        )
        result = await orch.dispatch(prompt="Hello", models=["slow_model"])
        resp = result.responses["slow_model"]
        assert resp.is_error
        assert "Timeout" in resp.error


class TestOrchestratorErrors:
    @pytest.mark.asyncio
    async def test_error_isolation(self) -> None:
        orch = CockpitOrchestrator(model_call_fn=_partial_error_call)
        result = await orch.dispatch(
            prompt="Hello",
            models=["good_model", "bad_model"],
        )
        assert result.responses["good_model"].content == "OK from good_model"
        assert result.responses["bad_model"].is_error
        assert "Bad model failed" in result.responses["bad_model"].error

    @pytest.mark.asyncio
    async def test_all_errors(self) -> None:
        orch = CockpitOrchestrator(model_call_fn=_error_call)
        result = await orch.dispatch(prompt="Hello", models=["m1", "m2"])
        assert all(r.is_error for r in result.responses.values())


class TestOrchestratorRoles:
    @pytest.mark.asyncio
    async def test_role_prompts_used(self) -> None:
        calls: list[tuple[str, str, str]] = []

        async def _capturing_call(
            model_id: str, prompt: str, system_context: str
        ) -> ModelResponse:
            calls.append((model_id, prompt, system_context))
            return ModelResponse(model_id=model_id, content="OK")

        orch = CockpitOrchestrator(model_call_fn=_capturing_call)
        await orch.dispatch(
            prompt="Build an API",
            models=["m1", "m2"],
            roles={"m1": ModelRole.CODER, "m2": ModelRole.REVIEWER},
        )

        assert len(calls) == 2
        # m1 should get coder prompt, m2 reviewer prompt
        m1_ctx = [c[2] for c in calls if c[0] == "m1"][0]
        m2_ctx = [c[2] for c in calls if c[0] == "m2"][0]
        assert "engineer" in m1_ctx.lower() or "code" in m1_ctx.lower()
        assert "review" in m2_ctx.lower()
