"""Multi-LLM workflow orchestration endpoints.

Endpoints:
    POST   /workflows          -- Create a workflow (chain or consensus)
    GET    /workflows          -- List user's workflows
    GET    /workflows/{id}     -- Get a workflow by ID
    POST   /workflows/{id}/run -- Execute a workflow
    GET    /workflows/{id}/runs -- List executions for a workflow
    DELETE /workflows/{id}     -- Delete a workflow
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.v1.auth import get_current_user
from app.core.cockpit.chain import ChainExecutor, ChainStep
from app.core.cockpit.consensus import ConsensusEngine

logger = structlog.get_logger()

router = APIRouter()


# ── In-memory stores (production: DB) ────────────────────────────────────

_workflows: dict[str, dict[str, Any]] = {}
_workflow_runs: dict[str, list[dict[str, Any]]] = {}


# ── Request / Response Schemas ───────────────────────────────────────────


class ChainStepSchema(BaseModel):
    """A single step in a chain workflow."""

    model_id: str = Field(..., min_length=1, description="Model to use for this step")
    instruction: str = Field(default="", description="What this step should do")
    system_context: str = Field(default="", description="System prompt override")


class CreateWorkflowRequest(BaseModel):
    """Create a new workflow."""

    name: str = Field(..., min_length=1, max_length=200, description="Workflow name")
    workflow_type: str = Field(
        ..., pattern=r"^(chain|consensus)$", description="Type: chain or consensus"
    )
    description: str = Field(default="", max_length=1000)
    # Chain-specific
    steps: list[ChainStepSchema] = Field(
        default_factory=list,
        description="Steps for chain workflows (ordered)",
    )
    # Consensus-specific
    models: list[str] = Field(
        default_factory=list,
        description="Models for consensus workflows",
    )
    system_context: str = Field(
        default="", description="Shared system context for all steps/models"
    )


class RunWorkflowRequest(BaseModel):
    """Execute a workflow with a prompt."""

    prompt: str = Field(..., min_length=1, max_length=10_000, description="The prompt to run")


class WorkflowResponse(BaseModel):
    """A workflow definition."""

    id: str
    name: str
    workflow_type: str
    description: str
    steps: list[dict[str, Any]]
    models: list[str]
    system_context: str
    user_id: str
    created_at: float
    updated_at: float


class ChainStepResultSchema(BaseModel):
    """Result of a single chain step."""

    step_index: int
    model_id: str
    instruction: str
    content: str
    input_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""


class ConsensusResponseSchema(BaseModel):
    """Result of consensus evaluation."""

    agreement_level: str
    agreement_score: float
    common_themes: list[str]
    differences: list[str]
    summary: str
    responses: dict[str, dict[str, Any]]


class WorkflowRunResponse(BaseModel):
    """Result of a workflow execution."""

    id: str
    workflow_id: str
    prompt: str
    workflow_type: str
    status: str  # "completed" | "failed"
    result: dict[str, Any]
    total_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    created_at: float


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(
    body: CreateWorkflowRequest,
    user: dict = Depends(get_current_user),
) -> WorkflowResponse:
    """Create a new workflow definition."""
    user_id = user["sub"]
    now = time.time()

    # Validate chain has steps
    if body.workflow_type == "chain" and not body.steps:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Chain workflows require at least one step",
        )

    # Validate consensus has models
    if body.workflow_type == "consensus" and len(body.models) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Consensus workflows require at least 2 models",
        )

    workflow_id = uuid.uuid4().hex[:16]
    workflow = {
        "id": workflow_id,
        "name": body.name,
        "workflow_type": body.workflow_type,
        "description": body.description,
        "steps": [s.model_dump() for s in body.steps],
        "models": body.models,
        "system_context": body.system_context,
        "user_id": user_id,
        "created_at": now,
        "updated_at": now,
    }
    _workflows[workflow_id] = workflow

    return WorkflowResponse(**workflow)


@router.get("", response_model=list[WorkflowResponse])
async def list_workflows(
    user: dict = Depends(get_current_user),
) -> list[WorkflowResponse]:
    """List all workflows for the current user."""
    user_id = user["sub"]
    results = [
        WorkflowResponse(**wf)
        for wf in _workflows.values()
        if wf["user_id"] == user_id
    ]
    return sorted(results, key=lambda w: w.created_at, reverse=True)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    user: dict = Depends(get_current_user),
) -> WorkflowResponse:
    """Get a workflow by ID."""
    user_id = user["sub"]
    wf = _workflows.get(workflow_id)
    if not wf or wf["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return WorkflowResponse(**wf)


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_workflow(
    workflow_id: str,
    user: dict = Depends(get_current_user),
) -> None:
    """Delete a workflow."""
    user_id = user["sub"]
    wf = _workflows.get(workflow_id)
    if not wf or wf["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    del _workflows[workflow_id]
    _workflow_runs.pop(workflow_id, None)


@router.post("/{workflow_id}/run", response_model=WorkflowRunResponse)
async def run_workflow(
    workflow_id: str,
    body: RunWorkflowRequest,
    user: dict = Depends(get_current_user),
) -> WorkflowRunResponse:
    """Execute a workflow with the given prompt.

    - Chain: runs steps sequentially, each step's output feeds the next.
    - Consensus: runs all models in parallel, analyzes agreement.
    """
    user_id = user["sub"]
    wf = _workflows.get(workflow_id)
    if not wf or wf["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    run_id = uuid.uuid4().hex[:16]
    now = time.time()

    try:
        if wf["workflow_type"] == "chain":
            result_data = await _run_chain(wf, body.prompt)
        else:
            result_data = await _run_consensus(wf, body.prompt)

        run_status = "completed"
    except Exception as exc:
        logger.exception("workflow_run_error", workflow_id=workflow_id)
        result_data = {
            "error": str(exc),
            "total_latency_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }
        run_status = "failed"

    run_record = {
        "id": run_id,
        "workflow_id": workflow_id,
        "prompt": body.prompt,
        "workflow_type": wf["workflow_type"],
        "status": run_status,
        "result": result_data,
        "total_latency_ms": result_data.get("total_latency_ms", 0.0),
        "total_prompt_tokens": result_data.get("total_prompt_tokens", 0),
        "total_completion_tokens": result_data.get("total_completion_tokens", 0),
        "created_at": now,
    }

    _workflow_runs.setdefault(workflow_id, []).append(run_record)

    return WorkflowRunResponse(**run_record)


@router.get("/{workflow_id}/runs", response_model=list[WorkflowRunResponse])
async def list_runs(
    workflow_id: str,
    user: dict = Depends(get_current_user),
) -> list[WorkflowRunResponse]:
    """List execution history for a workflow."""
    user_id = user["sub"]
    wf = _workflows.get(workflow_id)
    if not wf or wf["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    runs = _workflow_runs.get(workflow_id, [])
    return [WorkflowRunResponse(**r) for r in reversed(runs)]


# ── Execution Helpers ────────────────────────────────────────────────────


async def _run_chain(wf: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Execute a chain workflow via ChainExecutor."""
    executor = ChainExecutor()
    steps = [
        ChainStep(
            model_id=s["model_id"],
            instruction=s.get("instruction", ""),
            system_context=s.get("system_context", ""),
        )
        for s in wf["steps"]
    ]

    result = await executor.execute(
        prompt=prompt,
        steps=steps,
        system_context=wf.get("system_context", ""),
    )

    step_results = [
        {
            "step_index": sr.step_index,
            "model_id": sr.model_id,
            "instruction": sr.instruction,
            "content": sr.response.content,
            "input_text": sr.input_text,
            "prompt_tokens": sr.response.prompt_tokens,
            "completion_tokens": sr.response.completion_tokens,
            "latency_ms": sr.response.latency_ms,
            "error": sr.response.error,
        }
        for sr in result.steps
    ]

    return {
        "final_output": result.final_output,
        "steps": step_results,
        "had_errors": result.had_errors,
        "total_latency_ms": result.total_latency_ms,
        "total_prompt_tokens": result.total_prompt_tokens,
        "total_completion_tokens": result.total_completion_tokens,
    }


async def _run_consensus(wf: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Execute a consensus workflow via ConsensusEngine."""
    engine = ConsensusEngine()

    result = await engine.evaluate(
        prompt=prompt,
        models=wf["models"],
        system_context=wf.get("system_context", ""),
    )

    responses = {
        model_id: {
            "content": resp.content,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_ms": resp.latency_ms,
            "error": resp.error,
        }
        for model_id, resp in result.responses.items()
    }

    return {
        "agreement_level": result.agreement_level,
        "agreement_score": result.agreement_score,
        "common_themes": list(result.common_themes),
        "differences": list(result.differences),
        "summary": result.summary,
        "responses": responses,
        "total_latency_ms": result.total_latency_ms,
        "total_prompt_tokens": sum(
            r.prompt_tokens for r in result.responses.values()
        ),
        "total_completion_tokens": sum(
            r.completion_tokens for r in result.responses.values()
        ),
    }
