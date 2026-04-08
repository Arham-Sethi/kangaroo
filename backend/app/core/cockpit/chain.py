"""Chain executor — sequential model pipeline (A -> B -> C).

Passes the output of one model as input to the next, building
on intermediate results. Useful for refine-then-review flows.

Example:
    Chain: GPT-4o (draft) -> Claude (review) -> Gemini (final polish)

Usage:
    chain = ChainExecutor(call_fn=my_llm_call)
    result = await chain.execute(
        prompt="Write a REST API",
        steps=[
            ChainStep(model_id="openai/gpt-4o", instruction="Write the initial draft"),
            ChainStep(model_id="anthropic/claude-sonnet-4", instruction="Review and improve"),
        ],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.core.cockpit.orchestrator import ModelCallFn, ModelResponse


@dataclass(frozen=True)
class ChainStep:
    """A single step in a chain pipeline.

    Attributes:
        model_id: Model to use for this step.
        instruction: What this step should do.
        system_context: Optional override for system prompt.
    """

    model_id: str
    instruction: str = ""
    system_context: str = ""


@dataclass(frozen=True)
class ChainStepResult:
    """Result of a single chain step.

    Attributes:
        step_index: Position in the chain (0-based).
        model_id: Which model produced this result.
        instruction: What the step was asked to do.
        response: The model's response.
        input_text: What was fed to this step.
    """

    step_index: int
    model_id: str
    instruction: str
    response: ModelResponse
    input_text: str


@dataclass(frozen=True)
class ChainResult:
    """Result of executing a complete chain.

    Attributes:
        prompt: Original prompt.
        steps: Results from each step.
        final_output: Content from the last step.
        total_latency_ms: Total wall-clock time.
        total_prompt_tokens: Total prompt tokens across all steps.
        total_completion_tokens: Total completion tokens.
    """

    prompt: str
    steps: tuple[ChainStepResult, ...]
    final_output: str
    total_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def had_errors(self) -> bool:
        return any(s.response.is_error for s in self.steps)


class ChainExecutor:
    """Executes a sequential chain of model calls.

    Each step receives the output of the previous step as additional
    context, building a pipeline of refinement.
    """

    def __init__(self, call_fn: ModelCallFn | None = None) -> None:
        self._call_fn = call_fn or _default_chain_call

    async def execute(
        self,
        prompt: str,
        steps: list[ChainStep],
        system_context: str = "",
    ) -> ChainResult:
        """Execute a chain of model calls sequentially.

        Args:
            prompt: The original user prompt.
            steps: Ordered list of chain steps.
            system_context: Base system context for all steps.

        Returns:
            ChainResult with all step results.
        """
        if not steps:
            return ChainResult(
                prompt=prompt,
                steps=(),
                final_output="",
                total_latency_ms=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
            )

        start = time.monotonic()
        step_results: list[ChainStepResult] = []
        current_input = prompt

        for idx, step in enumerate(steps):
            # Build the input for this step
            if idx > 0:
                # Include previous output as context
                prev_output = step_results[-1].response.content
                step_input = (
                    f"Previous step output:\n{prev_output}\n\n"
                    f"Your task: {step.instruction}\n\n"
                    f"Original prompt: {prompt}"
                )
            else:
                step_input = (
                    f"{step.instruction}\n\n{prompt}" if step.instruction else prompt
                )

            # Build system context
            ctx = step.system_context or system_context

            # Call the model
            response = await self._call_fn(step.model_id, step_input, ctx)

            step_results.append(
                ChainStepResult(
                    step_index=idx,
                    model_id=step.model_id,
                    instruction=step.instruction,
                    response=response,
                    input_text=step_input,
                )
            )

            # If this step errored, stop the chain
            if response.is_error:
                break

        elapsed_ms = (time.monotonic() - start) * 1000
        final_output = step_results[-1].response.content if step_results else ""

        return ChainResult(
            prompt=prompt,
            steps=tuple(step_results),
            final_output=final_output,
            total_latency_ms=round(elapsed_ms, 2),
            total_prompt_tokens=sum(s.response.prompt_tokens for s in step_results),
            total_completion_tokens=sum(s.response.completion_tokens for s in step_results),
        )


async def _default_chain_call(
    model_id: str, prompt: str, system_context: str
) -> ModelResponse:
    """Default chain call — routes to the real LLM client."""
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
        )
    except LLMError as exc:
        return ModelResponse(
            model_id=model_id,
            error=str(exc),
        )
