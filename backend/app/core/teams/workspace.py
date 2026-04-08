"""Team workspace operations.

Manages team CRUD, member invitations, role changes, and removal.
RBAC hierarchy: owner > admin > member > viewer.

All mutations are immutable-style -- we never modify inputs, only produce
new query results. The actual database writes happen through SQLAlchemy
async sessions; callers are responsible for committing.

Usage:
    svc = TeamService(db)
    team = await svc.create_team(owner=user, name="Acme AI")
    await svc.invite_member(team_id=team.id, email="bob@acme.com", inviter=user)
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.db import Team, TeamMember, TeamRole, User

logger = structlog.get_logger()


# ── Tier Limits ──────────────────────────────────────────────────────────────

TIER_MAX_SEATS: dict[str, int] = {
    "free": 0,
    "pro": 4,
    "pro_team": 50,
    "enterprise": 500,
}


# ── Exceptions ───────────────────────────────────────────────────────────────


class TeamError(Exception):
    """Base exception for team operations."""


class TeamNotFoundError(TeamError):
    """Team does not exist or user has no access."""


class TeamPermissionError(TeamError):
    """User lacks required role for this operation."""


class TeamLimitError(TeamError):
    """Team or tier limit reached."""


class MemberExistsError(TeamError):
    """User is already a member of this team."""


# ── Helper ───────────────────────────────────────────────────────────────────


def _slugify(name: str) -> str:
    """Convert team name to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    return slug[:180] + "-" + uuid.uuid4().hex[:6]


ROLE_HIERARCHY: dict[str, int] = {
    TeamRole.OWNER.value: 4,
    TeamRole.ADMIN.value: 3,
    TeamRole.MEMBER.value: 2,
    TeamRole.VIEWER.value: 1,
}


def _has_role(member_role: str, required_role: str) -> bool:
    """Check if member_role is >= required_role in hierarchy."""
    return ROLE_HIERARCHY.get(member_role, 0) >= ROLE_HIERARCHY.get(required_role, 0)


# ── Service ──────────────────────────────────────────────────────────────────


class TeamService:
    """Stateless service for team workspace operations.

    Each method takes an AsyncSession and performs atomic queries.
    Callers are responsible for committing the transaction.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ── Create ────────────────────────────────────────────────────────────

    async def create_team(
        self,
        owner: User,
        name: str,
        description: str = "",
    ) -> Team:
        """Create a new team with the user as owner.

        Raises:
            TeamLimitError: If user's tier doesn't allow teams.
        """
        max_seats = TIER_MAX_SEATS.get(owner.subscription_tier, 0)
        if max_seats == 0:
            raise TeamLimitError(
                f"Tier '{owner.subscription_tier}' does not include team features. "
                "Upgrade to Pro or higher."
            )

        team = Team(
            name=name,
            slug=_slugify(name),
            description=description,
            max_members=max_seats,
            settings={},
            is_active=True,
        )
        self._db.add(team)
        await self._db.flush()

        membership = TeamMember(
            team_id=team.id,
            user_id=owner.id,
            role=TeamRole.OWNER.value,
            invited_by=None,
        )
        self._db.add(membership)
        await self._db.flush()

        await logger.ainfo(
            "team_created", team_id=str(team.id), owner_id=str(owner.id), name=name,
        )
        return team

    # ── List ──────────────────────────────────────────────────────────────

    async def list_user_teams(self, user_id: uuid.UUID) -> list[Team]:
        """List all teams the user belongs to."""
        result = await self._db.execute(
            select(Team)
            .join(TeamMember, TeamMember.team_id == Team.id)
            .where(
                TeamMember.user_id == user_id,
                Team.deleted_at.is_(None),
                Team.is_active.is_(True),
            )
            .order_by(Team.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_team(self, team_id: uuid.UUID, user_id: uuid.UUID) -> Team:
        """Get a team by ID, verifying user membership.

        Raises:
            TeamNotFoundError: If team doesn't exist or user is not a member.
        """
        result = await self._db.execute(
            select(Team)
            .join(TeamMember, TeamMember.team_id == Team.id)
            .where(
                Team.id == team_id,
                Team.deleted_at.is_(None),
                TeamMember.user_id == user_id,
            )
        )
        team = result.scalar_one_or_none()
        if team is None:
            raise TeamNotFoundError("Team not found or you are not a member.")
        return team

    # ── Members ───────────────────────────────────────────────────────────

    async def list_members(
        self, team_id: uuid.UUID, user_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """List all members of a team.

        Raises:
            TeamNotFoundError: If team doesn't exist or user has no access.
        """
        await self.get_team(team_id, user_id)

        result = await self._db.execute(
            select(TeamMember, User)
            .join(User, User.id == TeamMember.user_id)
            .where(TeamMember.team_id == team_id)
            .order_by(TeamMember.created_at)
        )
        rows = result.all()

        return [
            {
                "user_id": str(member.user_id),
                "email": user.email,
                "display_name": user.display_name,
                "role": member.role,
                "invited_by": str(member.invited_by) if member.invited_by else None,
                "joined_at": member.created_at.isoformat() if member.created_at else None,
            }
            for member, user in rows
        ]

    async def invite_member(
        self,
        team_id: uuid.UUID,
        email: str,
        inviter: User,
        role: str = TeamRole.MEMBER.value,
    ) -> TeamMember:
        """Invite a user to a team by email.

        Raises:
            TeamPermissionError: If inviter lacks admin role.
            TeamLimitError: If team is at capacity.
            MemberExistsError: If user is already a member.
            TeamNotFoundError: If invitee user not found.
        """
        inviter_member = await self._get_membership(team_id, inviter.id)
        if inviter_member is None or not _has_role(inviter_member.role, TeamRole.ADMIN.value):
            raise TeamPermissionError("Only owners and admins can invite members.")

        team = await self.get_team(team_id, inviter.id)
        member_count = await self._count_members(team_id)
        if member_count >= team.max_members:
            raise TeamLimitError(
                f"Team has reached its maximum of {team.max_members} members."
            )

        result = await self._db.execute(
            select(User).where(
                User.email == email.lower().strip(),
                User.deleted_at.is_(None),
            )
        )
        invitee = result.scalar_one_or_none()
        if invitee is None:
            raise TeamNotFoundError(f"No user found with email '{email}'.")

        existing = await self._get_membership(team_id, invitee.id)
        if existing is not None:
            raise MemberExistsError("User is already a member of this team.")

        if role == TeamRole.OWNER.value:
            raise TeamPermissionError("Cannot assign owner role via invitation.")

        membership = TeamMember(
            team_id=team_id,
            user_id=invitee.id,
            role=role,
            invited_by=inviter.id,
        )
        self._db.add(membership)
        await self._db.flush()

        await logger.ainfo(
            "team_member_invited",
            team_id=str(team_id),
            invitee_id=str(invitee.id),
            inviter_id=str(inviter.id),
        )
        return membership

    async def change_role(
        self,
        team_id: uuid.UUID,
        target_user_id: uuid.UUID,
        new_role: str,
        actor: User,
    ) -> None:
        """Change a member's role.

        Raises:
            TeamPermissionError: If actor lacks permission.
            TeamNotFoundError: If target is not a member.
        """
        actor_member = await self._get_membership(team_id, actor.id)
        if actor_member is None or not _has_role(actor_member.role, TeamRole.ADMIN.value):
            raise TeamPermissionError("Only owners and admins can change roles.")

        target_member = await self._get_membership(team_id, target_user_id)
        if target_member is None:
            raise TeamNotFoundError("Target user is not a member of this team.")

        if target_member.role == TeamRole.OWNER.value and actor_member.role != TeamRole.OWNER.value:
            raise TeamPermissionError("Only the owner can modify the owner role.")

        if new_role == TeamRole.OWNER.value:
            raise TeamPermissionError("Cannot assign owner role. Use ownership transfer.")

        if new_role == TeamRole.ADMIN.value and actor_member.role != TeamRole.OWNER.value:
            raise TeamPermissionError("Only the owner can promote to admin.")

        await self._db.execute(
            update(TeamMember)
            .where(TeamMember.team_id == team_id, TeamMember.user_id == target_user_id)
            .values(role=new_role)
        )

    async def remove_member(
        self,
        team_id: uuid.UUID,
        target_user_id: uuid.UUID,
        actor: User,
    ) -> None:
        """Remove a member from a team.

        Raises:
            TeamPermissionError: If actor lacks permission.
            TeamNotFoundError: If target is not a member.
        """
        actor_member = await self._get_membership(team_id, actor.id)
        if actor_member is None:
            raise TeamNotFoundError("You are not a member of this team.")

        is_self_removal = target_user_id == actor.id
        if not is_self_removal:
            if not _has_role(actor_member.role, TeamRole.ADMIN.value):
                raise TeamPermissionError("Only owners and admins can remove members.")

        target_member = await self._get_membership(team_id, target_user_id)
        if target_member is None:
            raise TeamNotFoundError("Target user is not a member of this team.")

        if target_member.role == TeamRole.OWNER.value:
            raise TeamPermissionError("Cannot remove the team owner.")

        await self._db.execute(
            delete(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == target_user_id,
            )
        )

    # ── Private Helpers ───────────────────────────────────────────────────

    async def _get_membership(
        self, team_id: uuid.UUID, user_id: uuid.UUID,
    ) -> TeamMember | None:
        """Get a specific membership record."""
        result = await self._db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id, TeamMember.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def _count_members(self, team_id: uuid.UUID) -> int:
        """Count members in a team."""
        result = await self._db.execute(
            select(func.count(TeamMember.id)).where(TeamMember.team_id == team_id)
        )
        return result.scalar_one()
