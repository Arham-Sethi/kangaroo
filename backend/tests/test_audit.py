"""Tests for the audit logging system.

Tests cover:
    - Writing audit entries
    - Querying by user
    - Querying by resource
    - Action counting
    - AuditEntry immutability
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security.audit import (
    ACTION_CONTEXT_SHIFT,
    ACTION_SESSION_CREATE,
    ACTION_USER_LOGIN,
    AuditEntry,
    AuditLogger,
    RESOURCE_CONTEXT,
    RESOURCE_SESSION,
    RESOURCE_USER,
)


class TestAuditLog:
    """Tests for AuditLogger.log()."""

    @pytest.mark.asyncio
    async def test_log_creates_entry(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        user_id = uuid.uuid4()
        entry = await audit.log(
            action=ACTION_USER_LOGIN,
            resource_type=RESOURCE_USER,
            user_id=user_id,
            ip_address="10.0.0.1",
        )
        assert isinstance(entry, AuditEntry)
        assert entry.action == ACTION_USER_LOGIN
        assert entry.resource_type == RESOURCE_USER
        assert entry.user_id == user_id
        assert entry.ip_address == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_log_with_metadata(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        entry = await audit.log(
            action=ACTION_CONTEXT_SHIFT,
            resource_type=RESOURCE_CONTEXT,
            metadata={"source_llm": "openai", "target_llm": "anthropic"},
        )
        assert entry.metadata["source_llm"] == "openai"
        assert entry.metadata["target_llm"] == "anthropic"

    @pytest.mark.asyncio
    async def test_log_with_resource_id(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        resource_id = uuid.uuid4()
        entry = await audit.log(
            action=ACTION_SESSION_CREATE,
            resource_type=RESOURCE_SESSION,
            resource_id=resource_id,
        )
        assert entry.resource_id == resource_id

    @pytest.mark.asyncio
    async def test_log_with_user_agent(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        entry = await audit.log(
            action=ACTION_USER_LOGIN,
            resource_type=RESOURCE_USER,
            user_agent="Mozilla/5.0",
        )
        assert entry.user_agent == "Mozilla/5.0"

    @pytest.mark.asyncio
    async def test_log_minimal(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        entry = await audit.log(
            action=ACTION_USER_LOGIN,
            resource_type=RESOURCE_USER,
        )
        assert entry.user_id is None
        assert entry.resource_id is None
        assert entry.ip_address is None
        assert entry.metadata == {}

    @pytest.mark.asyncio
    async def test_log_has_timestamp(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        entry = await audit.log(
            action=ACTION_USER_LOGIN,
            resource_type=RESOURCE_USER,
        )
        assert entry.created_at is not None

    @pytest.mark.asyncio
    async def test_log_has_unique_id(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        e1 = await audit.log(action="a", resource_type="x")
        e2 = await audit.log(action="b", resource_type="y")
        assert e1.id != e2.id


class TestAuditQuery:
    """Tests for querying audit logs."""

    @pytest.mark.asyncio
    async def test_get_user_log(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        user_id = uuid.uuid4()
        await audit.log(action="a.one", resource_type="x", user_id=user_id)
        await audit.log(action="a.two", resource_type="x", user_id=user_id)
        await audit.log(action="a.three", resource_type="x", user_id=uuid.uuid4())

        entries = await audit.get_user_log(user_id)
        assert len(entries) == 2
        assert all(e.user_id == user_id for e in entries)

    @pytest.mark.asyncio
    async def test_get_user_log_with_action_filter(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        user_id = uuid.uuid4()
        await audit.log(action="session.create", resource_type="session", user_id=user_id)
        await audit.log(action="context.shift", resource_type="context", user_id=user_id)

        entries = await audit.get_user_log(user_id, action_filter="session.create")
        assert len(entries) == 1
        assert entries[0].action == "session.create"

    @pytest.mark.asyncio
    async def test_get_user_log_pagination(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        user_id = uuid.uuid4()
        for i in range(5):
            await audit.log(action=f"action.{i}", resource_type="x", user_id=user_id)

        page1 = await audit.get_user_log(user_id, limit=2, offset=0)
        page2 = await audit.get_user_log(user_id, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2

    @pytest.mark.asyncio
    async def test_get_resource_log(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        resource_id = uuid.uuid4()
        await audit.log(
            action="session.create",
            resource_type="session",
            resource_id=resource_id,
        )
        await audit.log(
            action="session.update",
            resource_type="session",
            resource_id=resource_id,
        )

        entries = await audit.get_resource_log("session", resource_id)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_count_user_actions(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        user_id = uuid.uuid4()
        await audit.log(action="context.shift", resource_type="context", user_id=user_id)
        await audit.log(action="context.shift", resource_type="context", user_id=user_id)
        await audit.log(action="user.login", resource_type="user", user_id=user_id)

        count = await audit.count_user_actions(user_id, "context.shift")
        assert count == 2

    @pytest.mark.asyncio
    async def test_count_user_actions_empty(self, db_session: AsyncSession) -> None:
        audit = AuditLogger(db_session)
        count = await audit.count_user_actions(uuid.uuid4(), "nonexistent")
        assert count == 0


class TestAuditEntry:
    """Tests for the AuditEntry dataclass."""

    def test_entry_is_frozen(self) -> None:
        entry = AuditEntry(
            id=uuid.uuid4(),
            user_id=None,
            action="test",
            resource_type="test",
            resource_id=None,
            ip_address=None,
            user_agent=None,
            metadata={},
            created_at=None,
        )
        with pytest.raises(AttributeError):
            entry.action = "hacked"  # type: ignore[misc]
