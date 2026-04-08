"""Context sharing between team members.

Allows team members to share sessions with their team, making them
visible in team views and team brain aggregation.

Usage:
    svc = SharingService(db)
    await svc.share_session(session_id, team_id, user_id)
    sessions = await svc.get_team_sessions(team_id, user_id)
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.db import Session, TeamMember, TeamRole

logger = structlog.get_logger()


class SharingError(Exception):
    """Base exception for sharing operations."""


class SharePermissionError(SharingError):
    """User lacks permission for this sharing operation."""


class ShareNotFoundError(SharingError):
    """Session or team not found."""


class SharingService:
    """Stateless service for session sharing within teams."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def share_session(
        self,
        session_id: uuid.UUID,
        team_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Share a session with a team.

        Sets session.team_id to make it visible to all team members.

        Raises:
            ShareNotFoundError: If session not found or user doesn't own it.
            SharePermissionError: If user isn't in the team or is a viewer.
        """
        result = await self._db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
                Session.deleted_at.is_(None),
            )
        )
        if result.scalar_one_or_none() is None:
            raise ShareNotFoundError("Session not found or you don't own it.")

        result = await self._db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id,
                TeamMember.user_id == user_id,
            )
        )
        membership = result.scalar_one_or_none()
        if membership is None:
            raise SharePermissionError("You are not a member of this team.")

        if membership.role == TeamRole.VIEWER.value:
            raise SharePermissionError("Viewers cannot share sessions.")

        await self._db.execute(
            update(Session).where(Session.id == session_id).values(team_id=team_id)
        )

        await logger.ainfo(
            "session_shared",
            session_id=str(session_id),
            team_id=str(team_id),
            user_id=str(user_id),
        )

    async def revoke_share(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Revoke sharing -- remove session from team.

        Only session owner or team admin can revoke.

        Raises:
            ShareNotFoundError: If session not found.
            SharePermissionError: If user lacks permission.
        """
        result = await self._db.execute(
            select(Session).where(Session.id == session_id, Session.deleted_at.is_(None))
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise ShareNotFoundError("Session not found.")

        if session.user_id != user_id:
            if session.team_id:
                result = await self._db.execute(
                    select(TeamMember).where(
                        TeamMember.team_id == session.team_id,
                        TeamMember.user_id == user_id,
                        TeamMember.role.in_([TeamRole.OWNER.value, TeamRole.ADMIN.value]),
                    )
                )
                if result.scalar_one_or_none() is None:
                    raise SharePermissionError(
                        "Only the session owner or team admin can revoke sharing."
                    )
            else:
                raise SharePermissionError("You don't own this session.")

        await self._db.execute(
            update(Session).where(Session.id == session_id).values(team_id=None)
        )

    async def get_team_sessions(
        self,
        team_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        """Get all sessions shared with a team.

        Raises:
            SharePermissionError: If user is not a member.
        """
        result = await self._db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == user_id,
            )
        )
        if result.scalar_one_or_none() is None:
            raise SharePermissionError("You are not a member of this team.")

        result = await self._db.execute(
            select(Session)
            .where(Session.team_id == team_id, Session.deleted_at.is_(None))
            .order_by(Session.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
