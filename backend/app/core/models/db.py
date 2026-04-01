"""SQLAlchemy ORM models for Kangaroo Shift.

These models define the database schema for the entire platform. Every table
is designed for a multi-tenant commercial SaaS with:
    - Subscription tiers gating feature access
    - Encryption at rest for all conversation content
    - Version branching for context history (git-like)
    - RBAC for team workspaces
    - Immutable audit logs for SOC2/GDPR compliance
    - Soft-delete support for data recovery and GDPR right-to-erasure

Naming conventions:
    - Table names: plural snake_case (users, sessions, contexts)
    - Primary keys: always 'id' as UUID
    - Foreign keys: '{referenced_table_singular}_id'
    - Timestamps: created_at, updated_at (UTC, auto-managed)
    - Soft-delete: deleted_at (nullable, NULL = active)
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Use PostgreSQL-native types when available, fall back to generic types.
# This allows the models to work with both PostgreSQL (production) and
# SQLite (testing). PostgreSQL gets JSONB (indexed, queryable); SQLite
# gets plain JSON (still stores/retrieves correctly).
JSONB = PG_JSONB().with_variant(JSON(), "sqlite")
UUID = PG_UUID


# ── Base ──────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Base class for all ORM models.

    Provides:
    - UUID primary key (globally unique, no integer enumeration attacks)
    - created_at / updated_at timestamps (auto-managed, always UTC)
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# ── Enums ─────────────────────────────────────────────────────────────────────


class SubscriptionTier(str, PyEnum):
    """Pricing tiers that gate feature access.

    FREE:       5 shifts/month, no local mode, no teams
    PRO:        Unlimited shifts, local mode, version history ($19/mo)
    TEAM:       All Pro + team workspaces, RBAC, audit logs ($49/user/mo)
    ENTERPRISE: All Team + SSO, dedicated infra, SLAs (custom pricing)
    """

    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class TeamRole(str, PyEnum):
    """RBAC roles within a team workspace."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class SourceLLM(str, PyEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENSOURCE = "opensource"
    UNKNOWN = "unknown"


# ── User ──────────────────────────────────────────────────────────────────────


class User(Base):
    """User account — the root of all ownership.

    Design decisions:
    - email is unique and indexed for login lookups
    - password_hash stores bcrypt output (never plaintext)
    - subscription_tier drives feature gating and billing
    - settings JSONB is a flexible bag for user preferences
    - shifts_this_month tracks usage for Free tier enforcement
    - deleted_at enables soft-delete for GDPR right-to-erasure
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320), unique=True, nullable=False, index=True,
    )
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    subscription_tier: Mapped[str] = mapped_column(
        String(20), default=SubscriptionTier.FREE.value, nullable=False,
    )
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    shifts_this_month: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    shifts_month_reset: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )
    api_keys: Mapped[list["APIKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )
    webhooks: Mapped[list["Webhook"]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )
    team_memberships: Mapped[list["TeamMember"]] = relationship(
        back_populates="user", cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_users_subscription_tier", "subscription_tier"),
        Index("ix_users_active", "is_active", "deleted_at"),
    )


# ── Session ───────────────────────────────────────────────────────────────────


class Session(Base):
    """A conversation session captured from an LLM.

    Represents one continuous conversation. A user might have hundreds
    of sessions. Each session has multiple context versions as the
    conversation grows and gets re-processed.

    The title is auto-generated from the first few messages but can
    be user-edited. Tags enable organization in the Context Library.
    """

    __tablename__ = "sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("teams.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), default="Untitled Session", nullable=False)
    source_llm: Mapped[str] = mapped_column(String(50), nullable=False)
    source_model: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tags: Mapped[dict] = mapped_column(JSONB, default=list, nullable=False)
    extra_data: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="sessions")
    team: Mapped["Team | None"] = relationship()
    contexts: Mapped[list["Context"]] = relationship(
        back_populates="session", cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_sessions_user_created", "user_id", "created_at"),
        Index("ix_sessions_user_active", "user_id", "deleted_at", "is_archived"),
        Index("ix_sessions_team", "team_id", "created_at"),
    )


# ── Context ───────────────────────────────────────────────────────────────────


class Context(Base):
    """A versioned, encrypted context snapshot.

    This is the actual product data — the UCS document, encrypted with
    AES-256-GCM and stored as a binary blob. The server never sees
    plaintext content when local encryption is enabled.

    Version branching works like git:
    - Each context has a version number (monotonically increasing)
    - parent_version_id links to the previous version (NULL = initial)
    - Branching: two contexts can share the same parent (fork point)
    - This enables "what if I had said X instead of Y?" workflows

    The encrypted_blob contains the serialized UCS JSON, encrypted.
    encryption_key_id identifies which key was used (for key rotation).
    """

    __tablename__ = "contexts"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    encrypted_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_key_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
        doc="Identifier for the encryption key used. Supports key rotation.",
    )
    blob_size_bytes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Size of encrypted blob for storage metering.",
    )
    compression_ratio: Mapped[float | None] = mapped_column(nullable=True)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contexts.id", ondelete="SET NULL"),
        nullable=True,
        doc="Parent context for version branching. NULL = root version.",
    )
    extra_data: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    is_current: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False,
        doc="Whether this is the latest version on its branch.",
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="contexts")
    parent_version: Mapped["Context | None"] = relationship(
        remote_side="Context.id",
    )

    __table_args__ = (
        UniqueConstraint("session_id", "version", name="uq_context_session_version"),
        Index("ix_contexts_session_current", "session_id", "is_current"),
        Index("ix_contexts_parent", "parent_version_id"),
    )


# ── APIKey ────────────────────────────────────────────────────────────────────


class APIKey(Base):
    """API key for SDK/CLI authentication.

    Design:
    - key_hash stores bcrypt hash of the API key (never plaintext)
    - key_prefix stores first 8 chars for identification in UI
    - last_used tracks activity for security monitoring
    - is_active allows instant revocation without deletion
    - Scopes enable fine-grained permissions (future: read-only keys)
    """

    __tablename__ = "api_keys"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    key_prefix: Mapped[str] = mapped_column(
        String(12), nullable=False,
        doc="First 8 chars of the key for display (e.g., 'ks_live_ab').",
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_used: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    scopes: Mapped[dict] = mapped_column(
        JSONB, default=list, nullable=False,
        doc="Permitted scopes: ['contexts:read', 'contexts:write', 'shifts:execute'].",
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_user_active", "user_id", "is_active"),
        Index("ix_api_keys_prefix", "key_prefix"),
    )


# ── Team ──────────────────────────────────────────────────────────────────────


class Team(Base):
    """Team workspace for collaborative context management.

    Teams are the $49/user/mo revenue driver. They provide:
    - Shared context library across team members
    - RBAC (owner > admin > member > viewer)
    - Audit trails for compliance
    - Usage analytics aggregated at team level
    """

    __tablename__ = "teams"

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(
        String(200), unique=True, nullable=False, index=True,
    )
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    max_members: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    settings: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    members: Mapped[list["TeamMember"]] = relationship(
        back_populates="team", cascade="all, delete-orphan",
    )


class TeamMember(Base):
    """Association between users and teams with role-based access."""

    __tablename__ = "team_members"

    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        String(20), default=TeamRole.MEMBER.value, nullable=False,
    )
    invited_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
    )

    # Relationships
    team: Mapped["Team"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="team_memberships")

    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_member"),
        Index("ix_team_members_user", "user_id"),
    )


# ── Webhook ───────────────────────────────────────────────────────────────────


class Webhook(Base):
    """Webhook subscription for event-driven integrations.

    Enables developers to build on top of Kangaroo Shift:
    - context.created, context.shifted, context.deleted
    - session.imported, session.exported
    - Secret is used to sign payloads (HMAC-SHA256) so recipients
      can verify the webhook came from us, not an attacker.
    """

    __tablename__ = "webhooks"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    events: Mapped[dict] = mapped_column(
        JSONB, nullable=False,
        doc="List of event types this webhook subscribes to.",
    )
    secret: Mapped[str] = mapped_column(
        String(128), nullable=False,
        doc="HMAC-SHA256 signing secret for payload verification.",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    failure_count: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False,
        doc="Consecutive delivery failures. Auto-disable at 10.",
    )
    last_triggered: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    last_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="webhooks")


# ── AuditLog ──────────────────────────────────────────────────────────────────


class AuditLog(Base):
    """Immutable audit log for security and compliance.

    Every sensitive action is logged: context access, shifts, exports,
    team changes, API key operations. This table is:
    - Append-only (no UPDATE or DELETE operations in application code)
    - Indexed for fast user/resource queries
    - Required for SOC2, GDPR, and enterprise compliance

    The metadata JSONB stores action-specific details (e.g., target LLM
    for a shift, old/new values for a settings change).
    """

    __tablename__ = "audit_logs"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action: Mapped[str] = mapped_column(
        String(100), nullable=False,
        doc="Action performed: 'context.shift', 'user.login', 'team.invite', etc.",
    )
    resource_type: Mapped[str] = mapped_column(
        String(50), nullable=False,
        doc="Type of resource affected: 'context', 'session', 'user', 'team'.",
    )
    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    extra_data: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_audit_logs_user_action", "user_id", "action"),
        Index("ix_audit_logs_resource", "resource_type", "resource_id"),
        Index("ix_audit_logs_created", "created_at"),
    )


# ── ShiftRecord ───────────────────────────────────────────────────────────────


class ShiftRecord(Base):
    """Record of every context shift — the core billable action.

    This table tracks every shift for:
    - Billing (count shifts per user per month)
    - Analytics (which LLM pairs are most popular?)
    - Quality monitoring (track success rates, latencies)
    - Debugging (when a shift fails, the full audit trail is here)
    """

    __tablename__ = "shift_records"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_context_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contexts.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_context_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contexts.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_llm: Mapped[str] = mapped_column(String(50), nullable=False)
    target_llm: Mapped[str] = mapped_column(String(50), nullable=False)
    source_model: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    target_model: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False,
        doc="Status: pending, processing, completed, failed.",
    )
    source_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    target_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compression_ratio: Mapped[float | None] = mapped_column(nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_data: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_shifts_user_created", "user_id", "created_at"),
        Index("ix_shifts_status", "status"),
        Index("ix_shifts_llm_pair", "source_llm", "target_llm"),
    )
