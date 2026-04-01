"""Initial database schema — all Phase 1 tables.

This migration creates the complete database schema for Kangaroo Shift:
    - 9 tables: users, sessions, contexts, api_keys, teams, team_members,
      webhooks, audit_logs, shift_records
    - updated_at trigger for automatic timestamp management
    - pgvector extension for future semantic search
    - Indexes optimized for the most common query patterns
    - CHECK constraints for data integrity at the database level

This is hand-crafted (not auto-generated) because Alembic's autogenerate
doesn't handle: PostgreSQL triggers, partial indexes, CHECK constraints,
or extension creation. For a commercial product, we need full control
over the schema.

Revision ID: 001
Revises: (initial)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Revision identifiers
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables, indexes, triggers, and extensions."""

    # ── Extensions ────────────────────────────────────────────────────────
    # pgvector: Enables vector similarity search for semantic context matching.
    # We'll use this in Phase 2 for embedding-based search.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # uuid-ossp: Provides uuid_generate_v4() as a database-level default.
    # While Python generates UUIDs, having DB-level generation is a safety net.
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # ── updated_at trigger function ───────────────────────────────────────
    # This PostgreSQL trigger automatically updates the updated_at column
    # whenever a row is modified. This is more reliable than application-level
    # onupdate because it works even for raw SQL updates and bulk operations.
    op.execute("""
        CREATE OR REPLACE FUNCTION trigger_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # ── Users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False, server_default=""),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("subscription_tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("shifts_this_month", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("shifts_month_reset", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # Unique email (only for active users — soft-deleted users don't block re-registration)
    op.create_index("ix_users_email_unique", "users", ["email"], unique=True, postgresql_where=sa.text("deleted_at IS NULL"))
    op.create_index("ix_users_subscription_tier", "users", ["subscription_tier"])
    op.create_index("ix_users_active", "users", ["is_active", "deleted_at"])
    op.create_index("ix_users_stripe", "users", ["stripe_customer_id"], unique=True, postgresql_where=sa.text("stripe_customer_id IS NOT NULL"))

    # CHECK: subscription tier must be a valid value
    op.execute("""
        ALTER TABLE users ADD CONSTRAINT ck_users_subscription_tier
        CHECK (subscription_tier IN ('free', 'pro', 'team', 'enterprise'))
    """)
    # CHECK: shifts count non-negative
    op.execute("ALTER TABLE users ADD CONSTRAINT ck_users_shifts_nonneg CHECK (shifts_this_month >= 0)")

    # Auto-update updated_at
    op.execute("""
        CREATE TRIGGER set_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()
    """)

    # ── Teams ─────────────────────────────────────────────────────────────
    op.create_table(
        "teams",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("max_members", sa.Integer, nullable=False, server_default=sa.text("10")),
        sa.Column("settings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_teams_slug_unique", "teams", ["slug"], unique=True, postgresql_where=sa.text("deleted_at IS NULL"))
    op.execute("ALTER TABLE teams ADD CONSTRAINT ck_teams_max_members CHECK (max_members >= 1)")
    op.execute("""
        CREATE TRIGGER set_teams_updated_at
        BEFORE UPDATE ON teams
        FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()
    """)

    # ── Team Members ──────────────────────────────────────────────────────
    op.create_table(
        "team_members",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("invited_by", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_unique_constraint("uq_team_member", "team_members", ["team_id", "user_id"])
    op.create_index("ix_team_members_user", "team_members", ["user_id"])
    op.create_index("ix_team_members_team_role", "team_members", ["team_id", "role"])
    op.execute("ALTER TABLE team_members ADD CONSTRAINT ck_team_members_role CHECK (role IN ('owner', 'admin', 'member', 'viewer'))")

    # ── Sessions ──────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("team_id", UUID(as_uuid=True), sa.ForeignKey("teams.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(500), nullable=False, server_default="Untitled Session"),
        sa.Column("source_llm", sa.String(50), nullable=False),
        sa.Column("source_model", sa.String(100), nullable=False, server_default=""),
        sa.Column("message_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("tags", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_archived", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
    op.create_index("ix_sessions_team_id", "sessions", ["team_id"])
    op.create_index("ix_sessions_user_created", "sessions", ["user_id", "created_at"])
    op.create_index("ix_sessions_user_active", "sessions", ["user_id", "deleted_at", "is_archived"])
    op.create_index("ix_sessions_source_llm", "sessions", ["source_llm"])
    op.execute("ALTER TABLE sessions ADD CONSTRAINT ck_sessions_counts CHECK (message_count >= 0 AND total_tokens >= 0)")
    op.execute("""
        CREATE TRIGGER set_sessions_updated_at
        BEFORE UPDATE ON sessions
        FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()
    """)

    # ── Contexts ──────────────────────────────────────────────────────────
    op.create_table(
        "contexts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("encrypted_blob", sa.LargeBinary, nullable=False),
        sa.Column("encryption_key_id", sa.String(255), nullable=False),
        sa.Column("blob_size_bytes", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("compression_ratio", sa.Float, nullable=True),
        sa.Column("parent_version_id", UUID(as_uuid=True), sa.ForeignKey("contexts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_current", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_unique_constraint("uq_context_session_version", "contexts", ["session_id", "version"])
    op.create_index("ix_contexts_session_id", "contexts", ["session_id"])
    op.create_index("ix_contexts_session_current", "contexts", ["session_id", "is_current"])
    op.create_index("ix_contexts_parent", "contexts", ["parent_version_id"])
    op.create_index("ix_contexts_encryption_key", "contexts", ["encryption_key_id"])
    op.execute("ALTER TABLE contexts ADD CONSTRAINT ck_contexts_version CHECK (version >= 1)")
    op.execute("ALTER TABLE contexts ADD CONSTRAINT ck_contexts_blob_size CHECK (blob_size_bytes >= 0)")

    # ── API Keys ──────────────────────────────────────────────────────────
    op.create_table(
        "api_keys",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_hash", sa.String(128), nullable=False),
        sa.Column("key_prefix", sa.String(12), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("last_used", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("scopes", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_api_keys_user_active", "api_keys", ["user_id", "is_active"])
    op.create_index("ix_api_keys_prefix", "api_keys", ["key_prefix"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)

    # ── Webhooks ──────────────────────────────────────────────────────────
    op.create_table(
        "webhooks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("events", JSONB, nullable=False),
        sa.Column("secret", sa.String(128), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("failure_count", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("last_triggered", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_status_code", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_webhooks_user_id", "webhooks", ["user_id"])
    op.create_index("ix_webhooks_active", "webhooks", ["is_active"], postgresql_where=sa.text("is_active = true"))
    op.execute("ALTER TABLE webhooks ADD CONSTRAINT ck_webhooks_failure_count CHECK (failure_count >= 0)")
    op.execute("""
        CREATE TRIGGER set_webhooks_updated_at
        BEFORE UPDATE ON webhooks
        FOR EACH ROW EXECUTE FUNCTION trigger_set_updated_at()
    """)

    # ── Audit Logs ────────────────────────────────────────────────────────
    # Append-only table. No UPDATE or DELETE triggers needed.
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50), nullable=False),
        sa.Column("resource_id", UUID(as_uuid=True), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        # No updated_at — audit logs are immutable
    )

    op.create_index("ix_audit_logs_user_action", "audit_logs", ["user_id", "action"])
    op.create_index("ix_audit_logs_resource", "audit_logs", ["resource_type", "resource_id"])
    op.create_index("ix_audit_logs_created", "audit_logs", ["created_at"])

    # ── Shift Records ─────────────────────────────────────────────────────
    op.create_table(
        "shift_records",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_context_id", UUID(as_uuid=True), sa.ForeignKey("contexts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("target_context_id", UUID(as_uuid=True), sa.ForeignKey("contexts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_llm", sa.String(50), nullable=False),
        sa.Column("target_llm", sa.String(50), nullable=False),
        sa.Column("source_model", sa.String(100), nullable=False, server_default=""),
        sa.Column("target_model", sa.String(100), nullable=False, server_default=""),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("source_tokens", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("target_tokens", sa.Integer, nullable=False, server_default=sa.text("0")),
        sa.Column("compression_ratio", sa.Float, nullable=True),
        sa.Column("processing_time_ms", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_shifts_user_created", "shift_records", ["user_id", "created_at"])
    op.create_index("ix_shifts_status", "shift_records", ["status"])
    op.create_index("ix_shifts_llm_pair", "shift_records", ["source_llm", "target_llm"])
    op.execute("""
        ALTER TABLE shift_records ADD CONSTRAINT ck_shifts_status
        CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
    """)


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("shift_records")
    op.drop_table("audit_logs")
    op.drop_table("webhooks")
    op.drop_table("api_keys")
    op.drop_table("contexts")
    op.drop_table("sessions")
    op.drop_table("team_members")
    op.drop_table("teams")
    op.drop_table("users")

    op.execute("DROP FUNCTION IF EXISTS trigger_set_updated_at()")
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')
    op.execute("DROP EXTENSION IF EXISTS vector")
