"""Tests for ChainExecutor — sequential model pipeline.

Tests cover:
    - Single step chain
    - Multi-step chain
    - Previous output passed to next step
    - Error stops chain
    - Empty steps
    - Token accumulation
    - Final output from last step
"""

import pytest

from app.core.cockpit.chain import (
    ChainExecutor,
    ChainResult,
    ChainStep,
    ChainStepResult,
)
from app.core.cockpit.orchestrator import ModelResponse


# -- Mock calls --------------------------------------------------------------


async def _mock_call(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    return ModelResponse(
        model_id=model_id,
        content=f"Output from {model_id}: processed",
        prompt_tokens=50,
        completion_tokens=20,
        latency_ms=10.0,
    )


async def _error_on_second(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    if "step2" in model_id:
        return ModelResponse(model_id=model_id, error="Step 2 failed")
    return ModelResponse(
        model_id=model_id,
        content=f"OK from {model_id}",
        prompt_tokens=30,
        completion_tokens=15,
    )


async def _echo_call(model_id: str, prompt: str, system_context: str) -> ModelResponse:
    """Echoes the prompt back so we can verify chaining."""
    return ModelResponse(
        model_id=model_id,
        content=f"Processed: {prompt[:100]}",
        prompt_tokens=len(prompt.split()),
        completion_tokens=10,
    )


# -- Tests -------------------------------------------------------------------


class TestChainExecution:
    @pytest.mark.asyncio
    async def test_single_step(self) -> None:
        chain = ChainExecutor(call_fn=_mock_call)
        result = await chain.execute(
            prompt="Hello",
            steps=[ChainStep(model_id="m1", instruction="Draft")],
        )
        assert isinstance(result, ChainResult)
        assert result.step_count == 1
        assert result.final_output == "Output from m1: processed"
        assert not result.had_errors

    @pytest.mark.asyncio
    async def test_multi_step(self) -> None:
        chain = ChainExecutor(call_fn=_mock_call)
        result = await chain.execute(
            prompt="Build API",
            steps=[
                ChainStep(model_id="m1", instruction="Draft"),
                ChainStep(model_id="m2", instruction="Review"),
                ChainStep(model_id="m3", instruction="Polish"),
            ],
        )
        assert result.step_count == 3
        assert result.final_output == "Output from m3: processed"

    @pytest.mark.asyncio
    async def test_empty_steps(self) -> None:
        chain = ChainExecutor(call_fn=_mock_call)
        result = await chain.execute(prompt="Hello", steps=[])
        assert result.step_count == 0
        assert result.final_output == ""

    @pytest.mark.asyncio
    async def test_token_accumulation(self) -> None:
        chain = ChainExecutor(call_fn=_mock_call)
        result = await chain.execute(
            prompt="Hello",
            steps=[
                ChainStep(model_id="m1"),
                ChainStep(model_id="m2"),
            ],
        )
        assert result.total_prompt_tokens == 100  # 50 * 2
        assert result.total_completion_tokens == 40  # 20 * 2


class TestChainPipeline:
    @pytest.mark.asyncio
    async def test_previous_output_in_next_input(self) -> None:
        chain = ChainExecutor(call_fn=_echo_call)
        result = await chain.execute(
            prompt="Original prompt",
            steps=[
                ChainStep(model_id="m1", instruction="First step"),
                ChainStep(model_id="m2", instruction="Second step"),
            ],
        )
        assert result.step_count == 2
        # Second step should reference previous output
        step2 = result.steps[1]
        assert "Previous step output" in step2.input_text


class TestChainErrors:
    @pytest.mark.asyncio
    async def test_error_stops_chain(self) -> None:
        chain = ChainExecutor(call_fn=_error_on_second)
        result = await chain.execute(
            prompt="Hello",
            steps=[
                ChainStep(model_id="step1"),
                ChainStep(model_id="step2"),
                ChainStep(model_id="step3"),
            ],
        )
        assert result.had_errors
        # Should stop at step 2, not reach step 3
        assert result.step_count == 2
        assert result.steps[1].response.is_error


class TestChainStepResult:
    @pytest.mark.asyncio
    async def test_step_result_fields(self) -> None:
        chain = ChainExecutor(call_fn=_mock_call)
        result = await chain.execute(
            prompt="Test",
            steps=[ChainStep(model_id="m1", instruction="Do something")],
        )
        step = result.steps[0]
        assert isinstance(step, ChainStepResult)
        assert step.step_index == 0
        assert step.model_id == "m1"
        assert step.instruction == "Do something"
        assert step.input_text  # Not empty
