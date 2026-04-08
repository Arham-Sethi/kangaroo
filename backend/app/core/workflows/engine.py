"""Workflow execution engine — generic pipeline for chain/consensus/dispatch.

Orchestrates complex multi-step workflows composed of model calls,
data transformations, and conditional branching.

This is the foundation that ChainExecutor and ConsensusEngine build on.

Usage:
    engine = WorkflowEngine()
    result = await engine.run(workflow)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class StepType(str, Enum):
    """Types of workflow steps."""

    MODEL_CALL = "model_call"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class WorkflowStep:
    """A single step in a workflow.

    Attributes:
        step_id: Unique identifier for this step.
        step_type: Type of step (model_call, parallel, etc.).
        config: Step-specific configuration.
        depends_on: List of step_ids this step depends on.
    """

    step_id: str
    step_type: StepType
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepResult:
    """Result of executing a workflow step.

    Attributes:
        step_id: Which step produced this result.
        output: The step's output data.
        latency_ms: Execution time.
        error: Error message if the step failed.
    """

    step_id: str
    output: Any = None
    latency_ms: float = 0.0
    error: str = ""

    @property
    def is_error(self) -> bool:
        return bool(self.error)


@dataclass(frozen=True)
class WorkflowResult:
    """Result of executing a complete workflow.

    Attributes:
        workflow_id: Identifier for this workflow execution.
        step_results: Map of step_id -> StepResult.
        final_output: Output from the last step.
        total_latency_ms: Total wall-clock time.
        success: Whether the workflow completed successfully.
    """

    workflow_id: str
    step_results: dict[str, StepResult]
    final_output: Any
    total_latency_ms: float
    success: bool


@dataclass(frozen=True)
class Workflow:
    """A complete workflow definition.

    Attributes:
        workflow_id: Unique identifier.
        steps: Ordered list of steps to execute.
        metadata: Additional workflow metadata.
    """

    workflow_id: str
    steps: tuple[WorkflowStep, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# Type for step executors
StepExecutor = Callable[[WorkflowStep, dict[str, StepResult]], Awaitable[StepResult]]


class WorkflowEngine:
    """Executes workflows by running steps in dependency order.

    Steps without dependencies run first. Steps with depends_on
    wait until their dependencies complete.
    """

    def __init__(self) -> None:
        self._executors: dict[StepType, StepExecutor] = {}

    def register_executor(
        self, step_type: StepType, executor: StepExecutor
    ) -> None:
        """Register an executor function for a step type."""
        self._executors[step_type] = executor

    async def run(self, workflow: Workflow) -> WorkflowResult:
        """Execute a workflow.

        Steps are executed in order. Each step receives the results
        of all previously completed steps.
        """
        start = time.monotonic()
        results: dict[str, StepResult] = {}
        final_output = None

        for step in workflow.steps:
            # Check dependencies
            for dep in step.depends_on:
                if dep not in results:
                    results[step.step_id] = StepResult(
                        step_id=step.step_id,
                        error=f"Missing dependency: {dep}",
                    )
                    continue
                if results[dep].is_error:
                    results[step.step_id] = StepResult(
                        step_id=step.step_id,
                        error=f"Dependency '{dep}' failed",
                    )
                    continue

            if step.step_id in results and results[step.step_id].is_error:
                continue

            executor = self._executors.get(step.step_type)
            if executor is None:
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    error=f"No executor for step type: {step.step_type.value}",
                )
                continue

            step_start = time.monotonic()
            try:
                result = await executor(step, results)
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    output=result.output,
                    latency_ms=round((time.monotonic() - step_start) * 1000, 2),
                )
                final_output = result.output
            except Exception as exc:
                results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    error=str(exc),
                    latency_ms=round((time.monotonic() - step_start) * 1000, 2),
                )

        elapsed = (time.monotonic() - start) * 1000
        success = all(not r.is_error for r in results.values())

        return WorkflowResult(
            workflow_id=workflow.workflow_id,
            step_results=results,
            final_output=final_output,
            total_latency_ms=round(elapsed, 2),
            success=success,
        )
