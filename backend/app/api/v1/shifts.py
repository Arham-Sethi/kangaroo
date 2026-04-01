"""Context shift (transfer between LLMs) endpoints.

POST /api/v1/shifts/execute  -- Shift context from one LLM format to another
POST /api/v1/shifts/preview  -- Preview what the shift would produce
GET  /api/v1/shifts/formats  -- List available input/output formats
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.adapters import AdapterRegistry, create_default_adapter_registry
from app.core.engine.ucs_generator import UCSGeneratorPipeline

router = APIRouter()

# Shared instances (stateless)
_pipeline = UCSGeneratorPipeline(enable_spacy=True)
_adapters: AdapterRegistry = create_default_adapter_registry()


# -- Request/Response schemas ------------------------------------------------


class ShiftExecuteRequest(BaseModel):
    """Execute a context shift from raw data to a target LLM format."""
    source_data: dict | list | str = Field(
        description="Raw conversation data in any supported format.",
    )
    target_format: str = Field(
        description="Target LLM format: openai, claude, or gemini.",
        pattern="^(openai|claude|gemini)$",
    )
    token_budget: int = Field(
        default=4000,
        ge=500,
        le=200000,
        description="Maximum tokens for the output context.",
    )


class ShiftPreviewRequest(BaseModel):
    """Preview a shift without full generation."""
    source_data: dict | list | str
    target_format: str = Field(pattern="^(openai|claude|gemini)$")


class ShiftExecuteResponse(BaseModel):
    """Full shift result with adapted messages."""
    target_format: str
    messages: list[dict[str, Any]]
    system_prompt: str
    token_estimate: int
    source_format: str
    entity_count: int
    summary_count: int
    compression_ratio: float
    model_suggestion: str


class ShiftPreviewResponse(BaseModel):
    """Preview of what a shift would produce (stats only, no messages)."""
    source_format: str
    message_count: int
    entity_count: int
    summary_count: int
    estimated_output_tokens: int
    target_format: str


class FormatsResponse(BaseModel):
    """Available input and output formats."""
    input_formats: list[str]
    output_formats: list[str]


# -- Endpoints ---------------------------------------------------------------


@router.post("/execute", response_model=ShiftExecuteResponse)
async def execute_shift(request: ShiftExecuteRequest) -> ShiftExecuteResponse:
    """Execute a full context shift.

    Takes raw conversation data, processes it through the full pipeline
    (parse -> extract -> summarize -> compress -> adapt), and returns
    messages ready for the target LLM.
    """
    try:
        # Process through pipeline with custom token budget
        pipeline = UCSGeneratorPipeline(
            target_tokens=request.token_budget,
            enable_spacy=True,
        )
        gen_result = pipeline.generate(request.source_data)

        # Adapt to target format
        adapted = _adapters.adapt(gen_result.ucs, target=request.target_format)

        return ShiftExecuteResponse(
            target_format=adapted.format_name,
            messages=adapted.messages,
            system_prompt=adapted.system_prompt,
            token_estimate=adapted.token_estimate,
            source_format=gen_result.conversation.source_format.value,
            entity_count=len(gen_result.ucs.entities),
            summary_count=len(gen_result.ucs.summaries),
            compression_ratio=gen_result.ucs.session_meta.compression_ratio,
            model_suggestion=adapted.metadata.get("model_suggestion", ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/preview", response_model=ShiftPreviewResponse)
async def preview_shift(request: ShiftPreviewRequest) -> ShiftPreviewResponse:
    """Preview a shift — returns stats without generating full output.

    Useful for showing the user what will happen before executing.
    """
    try:
        gen_result = _pipeline.generate(request.source_data)

        # Estimate output tokens
        adapted = _adapters.adapt(gen_result.ucs, target=request.target_format)

        return ShiftPreviewResponse(
            source_format=gen_result.conversation.source_format.value,
            message_count=gen_result.ucs.session_meta.message_count,
            entity_count=len(gen_result.ucs.entities),
            summary_count=len(gen_result.ucs.summaries),
            estimated_output_tokens=adapted.token_estimate,
            target_format=request.target_format,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.get("/formats", response_model=FormatsResponse)
async def list_formats() -> FormatsResponse:
    """List all available input and output formats."""
    return FormatsResponse(
        input_formats=["openai", "openai_export", "claude", "gemini", "generic"],
        output_formats=_adapters.available_formats,
    )
