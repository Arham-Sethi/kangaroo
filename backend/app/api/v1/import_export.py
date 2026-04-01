"""Conversation import/export endpoints.

POST /api/v1/import/text    -- Paste raw text
POST /api/v1/import/json    -- Upload JSON data
POST /api/v1/import/file    -- Upload a file
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.connectors.manual_import import ManualImportConnector

router = APIRouter()

# Shared connector instance (stateless, safe to reuse)
_connector = ManualImportConnector(enable_spacy=True)


# -- Request/Response schemas ------------------------------------------------


class TextImportRequest(BaseModel):
    """Import from pasted text."""
    content: str = Field(min_length=1, max_length=5_000_000)


class JsonImportRequest(BaseModel):
    """Import from JSON data."""
    data: dict | list


class ImportResponse(BaseModel):
    """Response from any import operation."""
    source_type: str
    detected_format: str
    original_size_bytes: int
    entity_count: int
    summary_count: int
    topic_count: int
    compression_ratio: float
    message_count: int
    ucs_version: str


# -- Helpers -----------------------------------------------------------------


def _to_response(result: Any) -> ImportResponse:
    """Convert ImportResult to API response."""
    gen = result.generation
    return ImportResponse(
        source_type=result.source_type,
        detected_format=result.detected_format,
        original_size_bytes=result.original_size_bytes,
        entity_count=len(gen.ucs.entities),
        summary_count=len(gen.ucs.summaries),
        topic_count=len(gen.ucs.topic_clusters),
        compression_ratio=gen.ucs.session_meta.compression_ratio,
        message_count=gen.ucs.session_meta.message_count,
        ucs_version=gen.ucs.version,
    )


# -- Endpoints ---------------------------------------------------------------


@router.post("/text", response_model=ImportResponse)
async def import_text(request: TextImportRequest) -> ImportResponse:
    """Import a conversation from pasted text.

    Accepts JSON strings, markdown conversations, or plain text.
    """
    try:
        result = _connector.import_text(request.content)
        return _to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/json", response_model=ImportResponse)
async def import_json(request: JsonImportRequest) -> ImportResponse:
    """Import a conversation from JSON data.

    Accepts any supported format: OpenAI, Claude, Gemini, ChatGPT export.
    """
    try:
        result = _connector.import_json(request.data)
        return _to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/file", response_model=ImportResponse)
async def import_file(file: UploadFile = File(...)) -> ImportResponse:
    """Import a conversation from an uploaded file (.json, .txt, .md)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        result = _connector.import_file(content, filename=file.filename or "")
        return _to_response(result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
