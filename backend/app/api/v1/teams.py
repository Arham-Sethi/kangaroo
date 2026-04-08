"""Team workspace and sharing endpoints.

Endpoints:
    POST   /teams                       -- Create a team
    GET    /teams                       -- List user's teams
    GET    /teams/{id}                  -- Get team details
    GET    /teams/{id}/members          -- List team members
    POST   /teams/{id}/invite           -- Invite a member
    PATCH  /teams/{id}/members/{uid}    -- Change member role
    DELETE /teams/{id}/members/{uid}    -- Remove a member
    POST   /teams/{id}/share/{sid}      -- Share a session with a team
    DELETE /teams/{id}/share/{sid}      -- Revoke session sharing
    GET    /teams/{id}/sessions         -- List team sessions
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.models.db import TeamRole, User
from app.core.teams.workspace import (
    MemberExistsError,
    TeamLimitError,
    TeamNotFoundError,
    TeamPermissionError,
    TeamService,
)
from app.core.teams.sharing import (
    ShareNotFoundError,
    SharePermissionError,
    SharingService,
)

logger = structlog.get_logger()
router = APIRouter()


# ── Schemas ──────────────────────────────────────────────────────────────────


class CreateTeamRequest(BaseModel):
    """Create a new team."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)


class InviteMemberRequest(BaseModel):
    """Invite a member by email."""

    email: EmailStr
    role: str = Field(default="member", pattern=r"^(admin|member|viewer)$")


class ChangeRoleRequest(BaseModel):
    """Change a member's role."""

    role: str = Field(..., pattern=r"^(admin|member|viewer)$")


class TeamResponse(BaseModel):
    """Team info response."""

    id: str
    name: str
    slug: str
    description: str
    max_members: int
    member_count: int
    created_at: str


class MemberResponse(BaseModel):
    """Team member info."""

    user_id: str
    email: str
    display_name: str
    role: str
    invited_by: str | None
    joined_at: str | None


class SessionShareRequest(BaseModel):
    """Share a session with a team."""
    pass  # session_id is in the URL path


# ── Helper ───────────────────────────────────────────────────────────────────


def _team_error_to_http(exc: Exception) -> HTTPException:
    """Convert team service errors to HTTP exceptions."""
    if isinstance(exc, TeamNotFoundError | ShareNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (TeamPermissionError, SharePermissionError)):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, TeamLimitError):
        return HTTPException(status_code=402, detail=str(exc))
    if isinstance(exc, MemberExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal server error")


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    body: CreateTeamRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TeamResponse:
    """Create a new team. Caller becomes the owner."""
    svc = TeamService(db)
    try:
        team = await svc.create_team(
            owner=current_user, name=body.name, description=body.description,
        )
        await db.commit()
        members = await svc.list_members(team.id, current_user.id)
        return TeamResponse(
            id=str(team.id),
            name=team.name,
            slug=team.slug,
            description=team.description,
            max_members=team.max_members,
            member_count=len(members),
            created_at=team.created_at.isoformat() if team.created_at else "",
        )
    except (TeamLimitError, TeamPermissionError) as exc:
        raise _team_error_to_http(exc)


@router.get("", response_model=list[TeamResponse])
async def list_teams(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[TeamResponse]:
    """List all teams the current user belongs to."""
    svc = TeamService(db)
    teams = await svc.list_user_teams(current_user.id)
    results = []
    for team in teams:
        members = await svc.list_members(team.id, current_user.id)
        results.append(TeamResponse(
            id=str(team.id),
            name=team.name,
            slug=team.slug,
            description=team.description,
            max_members=team.max_members,
            member_count=len(members),
            created_at=team.created_at.isoformat() if team.created_at else "",
        ))
    return results


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TeamResponse:
    """Get a team by ID."""
    svc = TeamService(db)
    try:
        team = await svc.get_team(team_id, current_user.id)
        members = await svc.list_members(team.id, current_user.id)
        return TeamResponse(
            id=str(team.id),
            name=team.name,
            slug=team.slug,
            description=team.description,
            max_members=team.max_members,
            member_count=len(members),
            created_at=team.created_at.isoformat() if team.created_at else "",
        )
    except TeamNotFoundError as exc:
        raise _team_error_to_http(exc)


@router.get("/{team_id}/members", response_model=list[MemberResponse])
async def list_members(
    team_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[MemberResponse]:
    """List all members of a team."""
    svc = TeamService(db)
    try:
        members = await svc.list_members(team_id, current_user.id)
        return [MemberResponse(**m) for m in members]
    except TeamNotFoundError as exc:
        raise _team_error_to_http(exc)


@router.post("/{team_id}/invite", response_model=MemberResponse, status_code=status.HTTP_201_CREATED)
async def invite_member(
    team_id: uuid.UUID,
    body: InviteMemberRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> MemberResponse:
    """Invite a user to a team by email."""
    svc = TeamService(db)
    try:
        membership = await svc.invite_member(
            team_id=team_id,
            email=body.email,
            inviter=current_user,
            role=body.role,
        )
        await db.commit()

        # Return member info
        members = await svc.list_members(team_id, current_user.id)
        invited = next(
            (m for m in members if m["user_id"] == str(membership.user_id)),
            None,
        )
        if invited:
            return MemberResponse(**invited)

        return MemberResponse(
            user_id=str(membership.user_id),
            email=body.email,
            display_name="",
            role=membership.role,
            invited_by=str(membership.invited_by) if membership.invited_by else None,
            joined_at=None,
        )
    except (TeamNotFoundError, TeamPermissionError, TeamLimitError, MemberExistsError) as exc:
        raise _team_error_to_http(exc)


@router.patch("/{team_id}/members/{user_id}")
async def change_member_role(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    body: ChangeRoleRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Change a team member's role."""
    svc = TeamService(db)
    try:
        await svc.change_role(
            team_id=team_id,
            target_user_id=user_id,
            new_role=body.role,
            actor=current_user,
        )
        await db.commit()
        return {"status": "ok", "user_id": str(user_id), "new_role": body.role}
    except (TeamNotFoundError, TeamPermissionError) as exc:
        raise _team_error_to_http(exc)


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def remove_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Remove a member from a team."""
    svc = TeamService(db)
    try:
        await svc.remove_member(
            team_id=team_id,
            target_user_id=user_id,
            actor=current_user,
        )
        await db.commit()
    except (TeamNotFoundError, TeamPermissionError) as exc:
        raise _team_error_to_http(exc)


# ── Sharing Endpoints ────────────────────────────────────────────────────────


@router.post("/{team_id}/share/{session_id}", status_code=status.HTTP_200_OK)
async def share_session(
    team_id: uuid.UUID,
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Share a session with a team."""
    svc = SharingService(db)
    try:
        await svc.share_session(session_id, team_id, current_user.id)
        await db.commit()
        return {"status": "shared", "session_id": str(session_id), "team_id": str(team_id)}
    except (ShareNotFoundError, SharePermissionError) as exc:
        raise _team_error_to_http(exc)


@router.delete("/{team_id}/share/{session_id}", status_code=status.HTTP_200_OK)
async def revoke_share(
    team_id: uuid.UUID,
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Revoke session sharing from a team."""
    svc = SharingService(db)
    try:
        await svc.revoke_share(session_id, current_user.id)
        await db.commit()
        return {"status": "revoked", "session_id": str(session_id)}
    except (ShareNotFoundError, SharePermissionError) as exc:
        raise _team_error_to_http(exc)


@router.get("/{team_id}/sessions")
async def list_team_sessions(
    team_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """List all sessions shared with a team."""
    svc = SharingService(db)
    try:
        sessions = await svc.get_team_sessions(team_id, current_user.id)
        return [
            {
                "id": str(s.id),
                "title": s.title,
                "source_llm": s.source_llm,
                "message_count": s.message_count,
                "total_tokens": s.total_tokens,
                "created_at": s.created_at.isoformat() if s.created_at else "",
                "updated_at": s.updated_at.isoformat() if s.updated_at else "",
            }
            for s in sessions
        ]
    except SharePermissionError as exc:
        raise _team_error_to_http(exc)
