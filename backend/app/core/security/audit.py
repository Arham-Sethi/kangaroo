"""Immutable audit logging for security and compliance.

Every sensitive action in Kangaroo Shift is logged to the audit_logs table.
This provides:

    - SOC2/GDPR compliance: full trail of who did what, when
    - Security monitoring: detect suspicious patterns (mass exports, etc.)
    - Debugging: trace issues back to specific actions
    - Enterprise features: audit reports for compliance officers

The audit logger is append-only. Once written, audit entries are NEVER
modified or deleted — not even by admins. This is enforced at the
application level (no update/delete methods exist) and should be
enforced at the database level via triggers in production.

Usage:
    from app.core.security.audit import AuditLogger

    audit = AuditLogger(db_session)
    await audit.log(
        user_id=user.id,
        action="context.shift",
        resource_type="context",
        resource_id=context.id,
        ip_address="10.0.0.1",
        metadata={"source_llm": "openai", "target_llm": "anthropic"},
    )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.sql import func

from app.core.models.db import AuditLog


# -- Action Constants --------------------------------------------------------

# Authentication
ACTION_USER_REGISTER = "user.register"
ACTION_USER_LOGIN = "user.login"
ACTION_USER_REFRESH = "user.refresh"

# Sessions
ACTION_SESSION_CREATE = "session.create"
ACTION_SESSION_UPDATE = "session.update"
ACTION_SESSION_ARCHIVE = "session.archive"
ACTION_SESSION_DELETE = "session.delete"

# Contexts
ACTION_CONTEXT_GENERATE = "context.generate"
ACTION_CONTEXT_SAVE = "context.save"
ACTION_CONTEXT_RETRIEVE = "context.retrieve"

# Shifts
ACTION_CONTEXT_SHIFT = "context.shift"

# Import/Export
ACTION_SESSION_IMPORT = "session.import"
ACTION_SESSION_EXPORT = "session.export"

# API Keys
ACTION_API_KEY_CREATE = "api_key.create"
ACTION_API_KEY_REVOKE = "api_key.revoke"

# Teams
ACTION_TEAM_CREATE = "team.create"
ACTION_TEAM_INVITE = "team.invite"
ACTION_TEAM_REMOVE = "team.remove"

# Webhooks
ACTION_WEBHOOK_CREATE = "webhook.create"
ACTION_WEBHOOK_DELETE = "webhook.delete"


# -- Resource Types ----------------------------------------------------------

RESOURCE_USER = "user"
RESOURCE_SESSION = "session"
RESOURCE_CONTEXT = "context"
RESOURCE_API_KEY = "api_key"
RESOURCE_TEAM = "team"
RESOURCE_WEBHOOK = "webhook"
RESOURCE_SHIFT = "shift"


# -- Result Dataclass -------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """Read-only representation of an audit log entry.

    Attributes:
        id: Unique entry ID.
        user_id: Who performed the action.
        action: What action was performed.
        resource_type: Type of resource affected.
        resource_id: Which resource was affected.
        ip_address: Client IP address.
        user_agent: Client user agent string.
        metadata: Action-specific details.
        created_at: When the action was performed.
    """

    id: uuid.UUID
    user_id: uuid.UUID | None
    action: str
    resource_type: str
    resource_id: uuid.UUID | None
    ip_address: str | None
    user_agent: str | None
    metadata: dict[str, Any]
    created_at: datetime


# -- Audit Logger -----------------------------------------------------------


class AuditLogger:
    """Append-only audit logger backed by the audit_logs database table.

    Thread-safe: each log call creates an independent database row.
    No shared mutable state.

    Usage:
        audit = AuditLogger(db)
        await audit.log(
            user_id=user.id,
            action="context.shift",
            resource_type="context",
        )
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize with a database session.

        Args:
            db: Async SQLAlchemy session for writing audit entries.
        """
        self._db = db

    async def log(
        self,
        action: str,
        resource_type: str,
        user_id: uuid.UUID | None = None,
        resource_id: uuid.UUID | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Write an audit log entry.

        This method ALWAYS succeeds or raises. It never silently drops
        audit entries — losing an audit entry is a compliance violation.

        Args:
            action: Action name (e.g., "context.shift").
            resource_type: Resource type (e.g., "context").
            user_id: Who performed the action.
            resource_id: Which resource was affected.
            ip_address: Client IP address.
            user_agent: Client user agent string.
            metadata: Additional action-specific details.

        Returns:
            AuditEntry representing the logged action.
        """
        entry = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            extra_data=metadata or {},
        )
        self._db.add(entry)
        await self._db.flush()

        return AuditEntry(
            id=entry.id,
            user_id=entry.user_id,
            action=entry.action,
            resource_type=entry.resource_type,
            resource_id=entry.resource_id,
            ip_address=entry.ip_address,
            user_agent=entry.user_agent,
            metadata=entry.extra_data,
            created_at=entry.created_at,
        )

    async def get_user_log(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
        action_filter: str | None = None,
    ) -> list[AuditEntry]:
        """Retrieve audit entries for a specific user.

        Args:
            user_id: User whose audit trail to retrieve.
            limit: Maximum entries to return.
            offset: Pagination offset.
            action_filter: Optional action type filter.

        Returns:
            List of AuditEntry objects, newest first.
        """
        query = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if action_filter:
            query = query.where(AuditLog.action == action_filter)

        result = await self._db.execute(query)
        rows = result.scalars().all()

        return [
            AuditEntry(
                id=row.id,
                user_id=row.user_id,
                action=row.action,
                resource_type=row.resource_type,
                resource_id=row.resource_id,
                ip_address=row.ip_address,
                user_agent=row.user_agent,
                metadata=row.extra_data,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def get_resource_log(
        self,
        resource_type: str,
        resource_id: uuid.UUID,
        limit: int = 50,
    ) -> list[AuditEntry]:
        """Retrieve audit entries for a specific resource.

        Args:
            resource_type: Type of resource.
            resource_id: Resource ID.
            limit: Maximum entries to return.

        Returns:
            List of AuditEntry objects, newest first.
        """
        query = (
            select(AuditLog)
            .where(
                AuditLog.resource_type == resource_type,
                AuditLog.resource_id == resource_id,
            )
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )

        result = await self._db.execute(query)
        rows = result.scalars().all()

        return [
            AuditEntry(
                id=row.id,
                user_id=row.user_id,
                action=row.action,
                resource_type=row.resource_type,
                resource_id=row.resource_id,
                ip_address=row.ip_address,
                user_agent=row.user_agent,
                metadata=row.extra_data,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def count_user_actions(
        self,
        user_id: uuid.UUID,
        action: str,
        since: datetime | None = None,
    ) -> int:
        """Count how many times a user performed an action.

        Useful for rate limiting and usage tracking.

        Args:
            user_id: User to count actions for.
            action: Action type to count.
            since: Only count actions after this timestamp.

        Returns:
            Number of matching audit entries.
        """
        query = (
            select(func.count())
            .select_from(AuditLog)
            .where(
                AuditLog.user_id == user_id,
                AuditLog.action == action,
            )
        )

        if since:
            query = query.where(AuditLog.created_at >= since)

        result = await self._db.execute(query)
        return result.scalar_one()
