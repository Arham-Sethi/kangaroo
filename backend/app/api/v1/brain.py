"""Brain endpoints — digest, knowledge gaps, and brain stats.

POST /api/v1/brain/digest    -- Generate a digest of recent sessions
POST /api/v1/brain/gaps      -- Detect knowledge gaps across sessions
GET  /api/v1/brain/stats     -- Get brain-level statistics

All endpoints require authentication. Data is scoped to the requesting user.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.brain.consolidator import MemoryConsolidator
from app.core.brain.digest import Digest, DigestConfig, DigestGenerator
from app.core.brain.gaps import GapConfig, GapDetector, GapReport
from app.core.database import get_db
from app.core.models.db import Context as ContextModel, Session as SessionModel, User
from app.core.storage.session_store import SessionStore

router = APIRouter()

_MASTER_KEY = "kangaroo-dev-master-key-change-in-prod"


# -- Schemas -----------------------------------------------------------------


class DigestRequest(BaseModel):
    """Request body for digest generation."""

    max_age_days: float = Field(
        default=1.0, ge=0.1, le=365, description="Include sessions from the last N days"
    )
    period_label: str = Field(default="Recent", max_length=100)


class DigestEntryResponse(BaseModel):
    category: str
    title: str
    detail: str
    importance: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DigestResponse(BaseModel):
    period_label: str
    session_count: int
    entries: list[DigestEntryResponse]
    new_entities: int
    decisions_made: int
    tasks_completed: int
    total_messages: int


class GapRequest(BaseModel):
    """Request body for gap detection."""

    min_mentions: int = Field(default=2, ge=1, le=20)
    stalled_days: float = Field(default=14.0, ge=1.0, le=365.0)


class GapResponse(BaseModel):
    gap_type: str
    title: str
    detail: str
    severity: str
    session_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class GapReportResponse(BaseModel):
    gaps: list[GapResponse]
    sessions_analyzed: int
    total_entities: int
    total_decisions: int
    total_tasks: int


class BrainStatsResponse(BaseModel):
    total_sessions: int
    total_contexts: int
    total_messages: int
    total_tokens: int
    active_sessions: int
    archived_sessions: int


# -- Helpers -----------------------------------------------------------------


async def _get_user_sessions(
    user: User,
    db: AsyncSession,
    max_age_days: float | None = None,
) -> list[SessionModel]:
    """Fetch user's sessions, optionally filtered by age."""
    query = select(SessionModel).where(
        SessionModel.user_id == user.id,
        SessionModel.is_archived == False,  # noqa: E712
    )

    if max_age_days is not None:
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        query = query.where(SessionModel.created_at >= cutoff)

    query = query.order_by(SessionModel.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


def _session_age_days(session: SessionModel) -> float:
    """Compute session age in fractional days."""
    now = datetime.now(timezone.utc)
    created = session.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (now - created).total_seconds() / 86400)


async def _load_session_ucs(
    sessions: list[SessionModel], db: AsyncSession
) -> list[Any]:
    """Load UCS data for sessions that have stored contexts.

    Returns UCS objects for sessions that have persisted contexts.
    For sessions without contexts, builds a minimal UCS from session metadata.
    """
    from app.core.models.ucs import (
        Preferences,
        SessionMeta,
        SourceLLM,
        UniversalContextSchema,
    )

    ucs_list = []
    for session in sessions:
        # Build minimal UCS from session metadata
        try:
            source = SourceLLM(session.source_llm)
        except ValueError:
            source = SourceLLM.UNKNOWN

        ucs = UniversalContextSchema(
            session_meta=SessionMeta(
                source_llm=source,
                source_model=session.source_model or "",
                message_count=session.message_count or 0,
                total_tokens=session.total_tokens or 0,
            ),
        )
        ucs_list.append(ucs)

    return ucs_list


# -- Endpoints ---------------------------------------------------------------


@router.post("/digest", response_model=DigestResponse)
async def generate_digest(
    body: DigestRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DigestResponse:
    """Generate a digest of recent sessions.

    Summarizes new entities, decisions, and tasks from the specified time period.
    """
    sessions = await _get_user_sessions(user, db, max_age_days=body.max_age_days)

    if not sessions:
        return DigestResponse(
            period_label=body.period_label,
            session_count=0,
            entries=[],
            new_entities=0,
            decisions_made=0,
            tasks_completed=0,
            total_messages=0,
        )

    ucs_list = await _load_session_ucs(sessions, db)

    generator = DigestGenerator()
    digest = generator.generate(ucs_list, period_label=body.period_label)

    return DigestResponse(
        period_label=digest.period_label,
        session_count=digest.session_count,
        entries=[
            DigestEntryResponse(
                category=e.category,
                title=e.title,
                detail=e.detail,
                importance=e.importance,
                metadata=e.metadata,
            )
            for e in digest.entries
        ],
        new_entities=digest.new_entities,
        decisions_made=digest.decisions_made,
        tasks_completed=digest.tasks_completed,
        total_messages=digest.total_messages,
    )


@router.post("/gaps", response_model=GapReportResponse)
async def detect_gaps(
    body: GapRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> GapReportResponse:
    """Detect knowledge gaps across all sessions.

    Identifies undecided topics, stalled tasks, and unclear entities.
    """
    sessions = await _get_user_sessions(user, db)

    if not sessions:
        return GapReportResponse(
            gaps=[],
            sessions_analyzed=0,
            total_entities=0,
            total_decisions=0,
            total_tasks=0,
        )

    ucs_list = await _load_session_ucs(sessions, db)
    ages = [_session_age_days(s) for s in sessions]

    detector = GapDetector(
        config=GapConfig(
            min_mentions_for_undecided=body.min_mentions,
            stalled_task_age_days=body.stalled_days,
        )
    )
    report = detector.detect(ucs_list, session_ages_days=ages)

    return GapReportResponse(
        gaps=[
            GapResponse(
                gap_type=g.gap_type,
                title=g.title,
                detail=g.detail,
                severity=g.severity,
                session_count=g.session_count,
                metadata=g.metadata,
            )
            for g in report.gaps
        ],
        sessions_analyzed=report.sessions_analyzed,
        total_entities=report.total_entities,
        total_decisions=report.total_decisions,
        total_tasks=report.total_tasks,
    )


@router.get("/stats", response_model=BrainStatsResponse)
async def brain_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> BrainStatsResponse:
    """Get brain-level statistics for the current user.

    Returns counts of sessions, contexts, messages, and tokens.
    """
    # Total sessions
    total_q = await db.execute(
        select(func.count(SessionModel.id)).where(
            SessionModel.user_id == user.id,
        )
    )
    total_sessions = total_q.scalar() or 0

    # Active sessions
    active_q = await db.execute(
        select(func.count(SessionModel.id)).where(
            SessionModel.user_id == user.id,
            SessionModel.is_archived == False,  # noqa: E712
        )
    )
    active_sessions = active_q.scalar() or 0

    # Archived sessions
    archived_sessions = total_sessions - active_sessions

    # Total messages and tokens
    msg_q = await db.execute(
        select(
            func.coalesce(func.sum(SessionModel.message_count), 0),
            func.coalesce(func.sum(SessionModel.total_tokens), 0),
        ).where(SessionModel.user_id == user.id)
    )
    row = msg_q.one()
    total_messages = row[0]
    total_tokens = row[1]

    # Total contexts
    ctx_q = await db.execute(
        select(func.count(ContextModel.id)).where(
            ContextModel.session_id.in_(
                select(SessionModel.id).where(SessionModel.user_id == user.id)
            )
        )
    )
    total_contexts = ctx_q.scalar() or 0

    return BrainStatsResponse(
        total_sessions=total_sessions,
        total_contexts=total_contexts,
        total_messages=total_messages,
        total_tokens=total_tokens,
        active_sessions=active_sessions,
        archived_sessions=archived_sessions,
    )
