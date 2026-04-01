"""Session CRUD and management endpoints.

Sessions represent captured conversations from LLMs. Users can:
    - List their sessions (with filtering and pagination)
    - Get a single session by ID
    - Update session title and tags
    - Archive/unarchive sessions
    - Soft-delete sessions

All endpoints require authentication via JWT access token.
All mutations are audit-logged for compliance.

Endpoints:
    GET    /api/v1/sessions          -- List user's sessions
    GET    /api/v1/sessions/{id}     -- Get session by ID
    POST   /api/v1/sessions          -- Create a new session
    PATCH  /api/v1/sessions/{id}     -- Update title/tags
    POST   /api/v1/sessions/{id}/archive   -- Archive session
    POST   /api/v1/sessions/{id}/unarchive -- Unarchive session
    DELETE /api/v1/sessions/{id}     -- Soft-delete session
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.models.db import Session, User
from app.core.security.audit import (
    ACTION_SESSION_ARCHIVE,
    ACTION_SESSION_CREATE,
    ACTION_SESSION_DELETE,
    ACTION_SESSION_UPDATE,
    AuditLogger,
    RESOURCE_SESSION,
)

router = APIRouter()
logger = structlog.get_logger()


# -- Request/Response Schemas ------------------------------------------------


class SessionCreate(BaseModel):
    """Create a new session."""

    title: str = Field(default="Untitled Session", max_length=500)
    source_llm: str = Field(max_length=50)
    source_model: str = Field(default="", max_length=100)
    message_count: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    """Update mutable session fields."""

    title: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = None


class SessionResponse(BaseModel):
    """Session in API responses."""

    id: uuid.UUID
    title: str
    source_llm: str
    source_model: str
    message_count: int
    total_tokens: int
    tags: list
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    """Paginated list of sessions."""

    sessions: list[SessionResponse]
    total: int
    limit: int
    offset: int


# -- Helpers -----------------------------------------------------------------


def _session_to_response(session: Session) -> SessionResponse:
    """Convert ORM Session to API response."""
    return SessionResponse(
        id=session.id,
        title=session.title,
        source_llm=session.source_llm,
        source_model=session.source_model,
        message_count=session.message_count,
        total_tokens=session.total_tokens,
        tags=session.tags if isinstance(session.tags, list) else [],
        is_archived=session.is_archived,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


async def _get_user_session(
    session_id: uuid.UUID,
    user: User,
    db: AsyncSession,
) -> Session:
    """Get a session owned by the user, or raise 404."""
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == user.id,
            Session.deleted_at.is_(None),
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return session


# -- Endpoints ---------------------------------------------------------------


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    archived: bool | None = Query(default=None),
    source_llm: str | None = Query(default=None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionListResponse:
    """List the current user's sessions with optional filtering.

    Args:
        limit: Max sessions per page (1-100).
        offset: Pagination offset.
        archived: Filter by archive status (None = all).
        source_llm: Filter by source LLM provider.
    """
    base_query = select(Session).where(
        Session.user_id == user.id,
        Session.deleted_at.is_(None),
    )

    if archived is not None:
        base_query = base_query.where(Session.is_archived == archived)
    if source_llm:
        base_query = base_query.where(Session.source_llm == source_llm)

    # Count total matching
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Fetch page
    page_query = (
        base_query
        .order_by(Session.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(page_query)
    sessions = result.scalars().all()

    return SessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Get a single session by ID."""
    session = await _get_user_session(session_id, user, db)
    return _session_to_response(session)


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Create a new session."""
    session = Session(
        user_id=user.id,
        title=body.title,
        source_llm=body.source_llm,
        source_model=body.source_model,
        message_count=body.message_count,
        total_tokens=body.total_tokens,
        tags=body.tags,
        extra_data=body.metadata,
    )
    db.add(session)
    await db.flush()

    # Audit log
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_SESSION_CREATE,
        resource_type=RESOURCE_SESSION,
        user_id=user.id,
        resource_id=session.id,
        ip_address=request.client.host if request.client else None,
        metadata={"title": body.title, "source_llm": body.source_llm},
    )

    await db.commit()
    await db.refresh(session)
    return _session_to_response(session)


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: uuid.UUID,
    body: SessionUpdate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Update a session's title and/or tags."""
    session = await _get_user_session(session_id, user, db)

    changes: dict = {}
    if body.title is not None:
        changes["title"] = {"old": session.title, "new": body.title}
        session.title = body.title
    if body.tags is not None:
        changes["tags"] = {"old": session.tags, "new": body.tags}
        session.tags = body.tags

    if not changes:
        return _session_to_response(session)

    # Audit log
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_SESSION_UPDATE,
        resource_type=RESOURCE_SESSION,
        user_id=user.id,
        resource_id=session.id,
        ip_address=request.client.host if request.client else None,
        metadata=changes,
    )

    await db.commit()
    await db.refresh(session)
    return _session_to_response(session)


@router.post("/{session_id}/archive", response_model=SessionResponse)
async def archive_session(
    session_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Archive a session (hide from default listing)."""
    session = await _get_user_session(session_id, user, db)

    if session.is_archived:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is already archived.",
        )

    session.is_archived = True

    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_SESSION_ARCHIVE,
        resource_type=RESOURCE_SESSION,
        user_id=user.id,
        resource_id=session.id,
        ip_address=request.client.host if request.client else None,
        metadata={"archived": True},
    )

    await db.commit()
    await db.refresh(session)
    return _session_to_response(session)


@router.post("/{session_id}/unarchive", response_model=SessionResponse)
async def unarchive_session(
    session_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    """Unarchive a session."""
    session = await _get_user_session(session_id, user, db)

    if not session.is_archived:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is not archived.",
        )

    session.is_archived = False

    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_SESSION_ARCHIVE,
        resource_type=RESOURCE_SESSION,
        user_id=user.id,
        resource_id=session.id,
        ip_address=request.client.host if request.client else None,
        metadata={"archived": False},
    )

    await db.commit()
    await db.refresh(session)
    return _session_to_response(session)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_session(
    session_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Soft-delete a session.

    The session is not actually removed from the database -- its
    deleted_at timestamp is set. This supports GDPR right-to-erasure
    recovery and data retention policies.
    """
    session = await _get_user_session(session_id, user, db)
    session.deleted_at = datetime.now(timezone.utc)

    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_SESSION_DELETE,
        resource_type=RESOURCE_SESSION,
        user_id=user.id,
        resource_id=session.id,
        ip_address=request.client.host if request.client else None,
        metadata={"title": session.title},
    )

    await db.commit()
