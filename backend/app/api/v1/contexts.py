"""Context CRUD and management endpoints.

POST /api/v1/contexts/generate     -- Generate UCS from raw data
POST /api/v1/contexts/generate/local -- Generate UCS locally (no network)
POST /api/v1/contexts/entities     -- Extract entities only
POST /api/v1/contexts/summarize    -- Summarize only
POST /api/v1/contexts/save         -- Save (encrypt + persist) a UCS to DB
GET  /api/v1/contexts/{session_id}/current -- Load current version
GET  /api/v1/contexts/{session_id}/versions -- List all versions
GET  /api/v1/contexts/{session_id}/versions/{context_id} -- Load specific version
DELETE /api/v1/contexts/versions/{context_id} -- Delete a version
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.engine.local_engine import LocalEngine, LocalProcessingConfig
from app.core.engine.ucs_generator import UCSGeneratorPipeline
from app.core.models.db import Session as SessionModel, User
from app.core.storage.session_store import SessionStore
from app.core.security.audit import (
    ACTION_CONTEXT_SAVE,
    ACTION_CONTEXT_RETRIEVE,
    AuditLogger,
    RESOURCE_CONTEXT,
)

router = APIRouter()

_pipeline = UCSGeneratorPipeline(enable_spacy=True)
_local_engine = LocalEngine(config=LocalProcessingConfig(enable_spacy=True))

# Master key for vault encryption (in production, from env/secret manager)
_MASTER_KEY = "kangaroo-dev-master-key-change-in-prod"


# -- Schemas -----------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Generate a UCS from raw conversation data."""
    data: dict | list | str = Field(
        description="Raw conversation in any supported format.",
    )
    token_budget: int = Field(
        default=4000, ge=500, le=200000,
    )


class GenerateResponse(BaseModel):
    """Full UCS generation result."""
    version: str
    source_format: str
    source_model: str
    message_count: int
    entity_count: int
    summary_count: int
    topic_count: int
    compression_ratio: float
    processing_mode: str
    processing_time_ms: float
    validation_warnings: list[str]
    entities: list[dict[str, Any]]
    knowledge_graph_nodes: int
    knowledge_graph_edges: int


class EntitiesRequest(BaseModel):
    """Extract entities from raw data."""
    data: dict | list | str


class EntitiesResponse(BaseModel):
    """Entity extraction result."""
    entity_count: int
    relationship_count: int
    entities: list[dict[str, Any]]
    knowledge_graph_nodes: int
    knowledge_graph_edges: int


class SummarizeRequest(BaseModel):
    """Summarize raw conversation data."""
    data: dict | list | str


class SummarizeResponse(BaseModel):
    """Summarization result."""
    message_summaries: int
    topic_summaries: int
    has_global_summary: bool
    global_summary: str | None
    topics: list[str]


# -- Endpoints ---------------------------------------------------------------


@router.post("/generate", response_model=GenerateResponse)
async def generate_context(request: GenerateRequest) -> GenerateResponse:
    """Generate a full UCS from raw conversation data.

    Runs the complete pipeline: parse -> extract -> summarize -> compress.
    """
    try:
        pipeline = UCSGeneratorPipeline(
            target_tokens=request.token_budget,
            enable_spacy=True,
        )
        result = pipeline.generate(request.data)
        ucs = result.ucs

        entities_list = [
            {
                "name": e.name,
                "type": e.type.value,
                "importance": e.importance,
                "aliases": list(e.aliases),
                "first_mentioned_at": e.first_mentioned_at,
            }
            for e in ucs.entities
        ]

        return GenerateResponse(
            version=ucs.version,
            source_format=result.stats.source_format,
            source_model=result.stats.source_model,
            message_count=ucs.session_meta.message_count,
            entity_count=len(ucs.entities),
            summary_count=len(ucs.summaries),
            topic_count=len(ucs.topic_clusters),
            compression_ratio=ucs.session_meta.compression_ratio,
            processing_mode=ucs.session_meta.processing_mode.value,
            processing_time_ms=result.stats.processing_time_ms,
            validation_warnings=list(result.stats.validation_warnings),
            entities=entities_list,
            knowledge_graph_nodes=len(ucs.knowledge_graph.nodes),
            knowledge_graph_edges=len(ucs.knowledge_graph.edges),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/generate/local", response_model=GenerateResponse)
async def generate_context_local(request: GenerateRequest) -> GenerateResponse:
    """Generate a UCS using local-only processing (no network calls).

    Privacy-first mode: everything runs on-device.
    """
    try:
        result = _local_engine.process(request.data)
        ucs = result.ucs

        entities_list = [
            {
                "name": e.name,
                "type": e.type.value,
                "importance": e.importance,
                "aliases": list(e.aliases),
            }
            for e in ucs.entities
        ]

        return GenerateResponse(
            version=ucs.version,
            source_format=result.stats.source_format,
            source_model=result.stats.source_model,
            message_count=ucs.session_meta.message_count,
            entity_count=len(ucs.entities),
            summary_count=len(ucs.summaries),
            topic_count=len(ucs.topic_clusters),
            compression_ratio=ucs.session_meta.compression_ratio,
            processing_mode=ucs.session_meta.processing_mode.value,
            processing_time_ms=result.stats.processing_time_ms,
            validation_warnings=list(result.stats.validation_warnings),
            entities=entities_list,
            knowledge_graph_nodes=len(ucs.knowledge_graph.nodes),
            knowledge_graph_edges=len(ucs.knowledge_graph.edges),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/entities", response_model=EntitiesResponse)
async def extract_entities(request: EntitiesRequest) -> EntitiesResponse:
    """Extract entities from raw conversation data (entities only, no UCS)."""
    try:
        result = _pipeline.generate(request.data)
        ucs = result.ucs

        entities_list = [
            {
                "name": e.name,
                "type": e.type.value,
                "importance": e.importance,
                "aliases": list(e.aliases),
                "relationships": [
                    {"target_id": str(r.target_id), "type": r.type.value}
                    for r in e.relationships
                ],
            }
            for e in ucs.entities
        ]

        return EntitiesResponse(
            entity_count=len(ucs.entities),
            relationship_count=result.stats.relationship_count,
            entities=entities_list,
            knowledge_graph_nodes=len(ucs.knowledge_graph.nodes),
            knowledge_graph_edges=len(ucs.knowledge_graph.edges),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_context(request: SummarizeRequest) -> SummarizeResponse:
    """Summarize raw conversation data (summaries only)."""
    try:
        result = _pipeline.generate(request.data)
        ucs = result.ucs

        from app.core.models.ucs import SummaryLevel
        global_summaries = [
            s for s in ucs.summaries if s.level == SummaryLevel.GLOBAL
        ]
        msg_summaries = [
            s for s in ucs.summaries if s.level == SummaryLevel.MESSAGE
        ]
        topic_summaries = [
            s for s in ucs.summaries if s.level == SummaryLevel.TOPIC
        ]

        return SummarizeResponse(
            message_summaries=len(msg_summaries),
            topic_summaries=len(topic_summaries),
            has_global_summary=len(global_summaries) > 0,
            global_summary=global_summaries[0].content if global_summaries else None,
            topics=[tc.label for tc in ucs.topic_clusters],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# -- Persistence Schemas -----------------------------------------------------


class SaveContextRequest(BaseModel):
    """Save a generated UCS to encrypted storage."""

    session_id: str = Field(description="Session UUID to save under.")
    data: dict | list | str = Field(description="Raw conversation data.")
    token_budget: int = Field(default=4000, ge=500, le=200000)
    metadata: dict[str, Any] | None = Field(
        default=None, description="Extra searchable metadata.",
    )


class SaveContextResponse(BaseModel):
    """Result of saving a context."""

    context_id: str
    version: int
    blob_size_bytes: int
    compression_ratio: float


class VersionResponse(BaseModel):
    """Single version metadata."""

    context_id: str
    version: int
    is_current: bool
    blob_size_bytes: int
    compression_ratio: float | None
    parent_version_id: str | None
    created_at: str
    metadata: dict[str, Any]


class VersionListResponse(BaseModel):
    """List of context versions."""

    versions: list[VersionResponse]
    total: int


class LoadContextResponse(BaseModel):
    """Loaded and decrypted UCS summary."""

    context_id: str
    version: str
    source_llm: str
    message_count: int
    entity_count: int
    summary_count: int
    decision_count: int
    task_count: int


# -- Persistence Helpers -----------------------------------------------------


from sqlalchemy import select  # noqa: E402


async def _verify_session_ownership(
    session_id: uuid.UUID,
    user: User,
    db: AsyncSession,
) -> SessionModel:
    """Verify user owns the session. Raises 404 if not found."""
    result = await db.execute(
        select(SessionModel).where(
            SessionModel.id == session_id,
            SessionModel.user_id == user.id,
            SessionModel.deleted_at.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


# -- Persistence Endpoints ---------------------------------------------------


@router.post("/save", status_code=201, response_model=SaveContextResponse)
async def save_context(
    body: SaveContextRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SaveContextResponse:
    """Generate UCS from raw data, encrypt, and persist to database.

    Combines generation + encryption + storage in one call.
    """
    try:
        sid = uuid.UUID(body.session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id UUID.")

    await _verify_session_ownership(sid, user, db)

    # Generate UCS
    try:
        pipeline = UCSGeneratorPipeline(
            target_tokens=body.token_budget,
            enable_spacy=True,
        )
        result = pipeline.generate(body.data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Encrypt + persist
    store = SessionStore(db, master_key=_MASTER_KEY)
    save_result = await store.save(
        session_id=sid,
        ucs=result.ucs,
        metadata=body.metadata,
    )

    # Audit
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_CONTEXT_SAVE,
        resource_type=RESOURCE_CONTEXT,
        user_id=user.id,
        resource_id=save_result.context_id,
        metadata={
            "session_id": str(sid),
            "version": save_result.version,
            "blob_size": save_result.blob_size_bytes,
        },
    )

    await db.commit()

    return SaveContextResponse(
        context_id=str(save_result.context_id),
        version=save_result.version,
        blob_size_bytes=save_result.blob_size_bytes,
        compression_ratio=save_result.compression_ratio,
    )


@router.get("/{session_id}/current", response_model=LoadContextResponse)
async def load_current_context(
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LoadContextResponse:
    """Load the current (latest) encrypted context for a session."""
    await _verify_session_ownership(session_id, user, db)

    store = SessionStore(db, master_key=_MASTER_KEY)
    ucs = await store.load(session_id)

    if ucs is None:
        raise HTTPException(status_code=404, detail="No saved context for this session.")

    # Audit
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_CONTEXT_RETRIEVE,
        resource_type=RESOURCE_CONTEXT,
        user_id=user.id,
        metadata={"session_id": str(session_id)},
    )
    await db.commit()

    return LoadContextResponse(
        context_id=str(session_id),
        version=ucs.version,
        source_llm=ucs.session_meta.source_llm.value,
        message_count=ucs.session_meta.message_count,
        entity_count=len(ucs.entities),
        summary_count=len(ucs.summaries),
        decision_count=len(ucs.decisions),
        task_count=len(ucs.tasks),
    )


@router.get("/{session_id}/versions", response_model=VersionListResponse)
async def list_context_versions(
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> VersionListResponse:
    """List all context versions for a session (metadata only)."""
    await _verify_session_ownership(session_id, user, db)

    store = SessionStore(db, master_key=_MASTER_KEY)
    versions = await store.list_versions(session_id, limit=limit, offset=offset)

    return VersionListResponse(
        versions=[
            VersionResponse(
                context_id=str(v.context_id),
                version=v.version,
                is_current=v.is_current,
                blob_size_bytes=v.blob_size_bytes,
                compression_ratio=v.compression_ratio,
                parent_version_id=str(v.parent_version_id) if v.parent_version_id else None,
                created_at=v.created_at.isoformat(),
                metadata=v.metadata,
            )
            for v in versions
        ],
        total=len(versions),
    )


@router.get(
    "/{session_id}/versions/{context_id}",
    response_model=LoadContextResponse,
)
async def load_context_version(
    session_id: uuid.UUID,
    context_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LoadContextResponse:
    """Load a specific context version by ID."""
    await _verify_session_ownership(session_id, user, db)

    store = SessionStore(db, master_key=_MASTER_KEY)
    ucs = await store.load(session_id, context_id=context_id)

    if ucs is None:
        raise HTTPException(status_code=404, detail="Context version not found.")

    return LoadContextResponse(
        context_id=str(context_id),
        version=ucs.version,
        source_llm=ucs.session_meta.source_llm.value,
        message_count=ucs.session_meta.message_count,
        entity_count=len(ucs.entities),
        summary_count=len(ucs.summaries),
        decision_count=len(ucs.decisions),
        task_count=len(ucs.tasks),
    )


@router.delete(
    "/versions/{context_id}",
    status_code=204,
    response_model=None,
)
async def delete_context_version(
    context_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a specific context version."""
    from app.core.storage.repository import ContextRepository

    repo = ContextRepository(db)
    record = await repo.load(context_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Context not found.")

    await _verify_session_ownership(record.session_id, user, db)

    deleted = await repo.delete(context_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Context not found.")
    await db.commit()
