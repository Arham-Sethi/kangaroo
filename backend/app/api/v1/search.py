"""Context search endpoints — keyword + semantic search across sessions.

GET  /api/v1/search           -- Keyword search across session metadata
POST /api/v1/search/semantic  -- Semantic (vector) search across sessions
GET  /api/v1/search/recall    -- Hybrid recall (keyword + semantic + recency)

All endpoints require authentication. Results are scoped to the
requesting user's sessions only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.brain.recall import (
    MemoryRecall,
    RecallConfig,
    RecallDocument,
    RecallHit,
)
from app.core.database import get_db
from app.core.models.db import Session as SessionModel, User
from app.core.storage.embeddings import EmbeddingEngine
from app.core.storage.search import SearchDocument, SearchEngine

router = APIRouter()

# Module-level singletons (reset in tests via app.dependency_overrides)
_search_engine = SearchEngine()
_embedding_engine = EmbeddingEngine()
_recall = MemoryRecall()


# -- Schemas -----------------------------------------------------------------


class SearchHitResponse(BaseModel):
    """A single search result."""

    id: str
    score: float
    snippet: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeywordSearchResponse(BaseModel):
    """Paginated keyword search response."""

    hits: list[SearchHitResponse]
    total: int
    query: str
    page: int
    page_size: int


class SemanticSearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)


class RecallRequest(BaseModel):
    """Request body for hybrid recall search."""

    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(default=20, ge=1, le=100)
    keyword_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)


class RecallHitResponse(BaseModel):
    """A single recall result."""

    document_id: str
    score: float
    keyword_score: float
    semantic_score: float
    recency_factor: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallSearchResponse(BaseModel):
    """Hybrid recall search response."""

    hits: list[RecallHitResponse]
    total: int
    query: str


# -- Helpers -----------------------------------------------------------------


async def _get_user_sessions(
    user: User, db: AsyncSession
) -> list[SessionModel]:
    """Fetch all sessions belonging to the current user."""
    result = await db.execute(
        select(SessionModel).where(
            SessionModel.user_id == user.id,
            SessionModel.is_archived == False,  # noqa: E712
        )
    )
    return list(result.scalars().all())


def _session_to_search_doc(session: SessionModel) -> SearchDocument:
    """Convert a DB session to a SearchDocument for keyword indexing."""
    parts = [session.title or ""]
    if session.tags:
        parts.extend(session.tags)

    return SearchDocument(
        id=str(session.id),
        content=" ".join(parts),
        metadata={
            "session_id": str(session.id),
            "title": session.title or "",
            "source_llm": session.source_llm or "",
        },
    )


def _session_to_recall_doc(session: SessionModel) -> RecallDocument:
    """Convert a DB session to a RecallDocument for hybrid search."""
    parts = [session.title or ""]
    if session.tags:
        parts.extend(session.tags)

    # Compute age in days from created_at
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    created = session.created_at
    if created.tzinfo is None:
        from datetime import timezone as tz

        created = created.replace(tzinfo=tz.utc)
    age_days = (now - created).total_seconds() / 86400

    return RecallDocument(
        id=str(session.id),
        text=" ".join(parts),
        age_days=max(0.0, age_days),
        metadata={
            "session_id": str(session.id),
            "title": session.title or "",
            "source_llm": session.source_llm or "",
        },
    )


# -- Endpoints ---------------------------------------------------------------


@router.get("", response_model=KeywordSearchResponse)
async def keyword_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KeywordSearchResponse:
    """Search sessions by keyword.

    Indexes session titles, tags, and summaries using BM25 scoring.
    Results are scoped to the authenticated user's sessions.
    """
    sessions = await _get_user_sessions(user, db)

    if not sessions:
        return KeywordSearchResponse(
            hits=[], total=0, query=q, page=page, page_size=page_size
        )

    # Build ephemeral index from user's sessions
    engine = SearchEngine()
    docs = [_session_to_search_doc(s) for s in sessions]
    engine.index(docs)

    result = engine.search(query=q, page=page, page_size=page_size)

    return KeywordSearchResponse(
        hits=[
            SearchHitResponse(
                id=hit.id, score=hit.score, snippet=hit.snippet, metadata=hit.metadata
            )
            for hit in result.hits
        ],
        total=result.total,
        query=result.query,
        page=result.page,
        page_size=result.page_size,
    )


@router.post("/semantic", response_model=KeywordSearchResponse)
async def semantic_search(
    body: SemanticSearchRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KeywordSearchResponse:
    """Search sessions by semantic similarity.

    Generates an embedding for the query and compares against
    session content using cosine similarity.
    """
    sessions = await _get_user_sessions(user, db)

    if not sessions:
        return KeywordSearchResponse(
            hits=[], total=0, query=body.query, page=1, page_size=body.top_k
        )

    # Build vocabulary from session texts
    engine = EmbeddingEngine()
    texts = []
    for s in sessions:
        parts = [s.title or ""]
        if s.tags:
            parts.extend(s.tags)
        texts.append(" ".join(parts))

    engine.build_vocabulary(texts)

    # Embed query and all docs
    query_emb = engine.embed(body.query)
    doc_tuples: list[tuple[str, tuple[float, ...], dict[str, Any]]] = []
    for session, text in zip(sessions, texts):
        emb = engine.embed(text)
        doc_tuples.append(
            (
                str(session.id),
                emb.vector,
                {
                    "session_id": str(session.id),
                    "title": session.title or "",
                    "source_llm": session.source_llm or "",
                },
            )
        )

    results = engine.search(query_emb.vector, doc_tuples, top_k=body.top_k)

    return KeywordSearchResponse(
        hits=[
            SearchHitResponse(
                id=hit.id, score=hit.score, snippet="", metadata=hit.metadata
            )
            for hit in results
        ],
        total=len(results),
        query=body.query,
        page=1,
        page_size=body.top_k,
    )


@router.post("/recall", response_model=RecallSearchResponse)
async def recall_search(
    body: RecallRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> RecallSearchResponse:
    """Hybrid recall search combining keywords, semantics, and recency.

    This is the primary "memory recall" endpoint — finds the most
    relevant past sessions using all available signals.
    """
    sessions = await _get_user_sessions(user, db)

    if not sessions:
        return RecallSearchResponse(hits=[], total=0, query=body.query)

    # Build recall documents
    recall_docs = [_session_to_recall_doc(s) for s in sessions]

    # Generate query vector for semantic component
    texts = [d.text for d in recall_docs]
    emb_engine = EmbeddingEngine()
    emb_engine.build_vocabulary(texts)
    query_emb = emb_engine.embed(body.query)

    # Attach vectors to recall docs
    docs_with_vectors = []
    for doc in recall_docs:
        vec = emb_engine.embed(doc.text).vector
        docs_with_vectors.append(
            RecallDocument(
                id=doc.id,
                text=doc.text,
                vector=vec,
                age_days=doc.age_days,
                metadata=doc.metadata,
            )
        )

    # Run hybrid recall
    recall = MemoryRecall(
        config=RecallConfig(
            keyword_weight=body.keyword_weight,
            semantic_weight=body.semantic_weight,
            max_results=body.max_results,
        )
    )

    hits = recall.search(
        query=body.query,
        documents=docs_with_vectors,
        query_vector=query_emb.vector,
    )

    return RecallSearchResponse(
        hits=[
            RecallHitResponse(
                document_id=h.document_id,
                score=h.score,
                keyword_score=h.keyword_score,
                semantic_score=h.semantic_score,
                recency_factor=h.recency_factor,
                metadata=h.metadata,
            )
            for h in hits
        ],
        total=len(hits),
        query=body.query,
    )
