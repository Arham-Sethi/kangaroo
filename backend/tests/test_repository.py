"""Tests for ContextRepository — async CRUD for the contexts table.

Tests cover:
    - Save a VaultEntry to DB
    - Load by ID
    - Load current version for a session
    - List versions (ordering, pagination)
    - Count versions
    - Delete a version
    - Version auto-incrementing (is_current management)
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.ucs import (
    SessionMeta,
    SourceLLM,
    UniversalContextSchema,
)
from app.core.storage.repository import ContextRepository, ContextRecord
from app.core.storage.vault import Vault


# -- Helpers -----------------------------------------------------------------

MASTER_KEY = "test-repo-master-key-32chars!!"


def _minimal_ucs() -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=5,
            total_tokens=800,
        ),
    )


def _make_vault_entry(session_id: uuid.UUID, version: int = 1):
    """Create a VaultEntry by encrypting a minimal UCS through the Vault."""
    vault = Vault(master_key=MASTER_KEY)
    return vault.store(
        _minimal_ucs(),
        session_id=session_id,
        version=version,
    )


# -- Tests -------------------------------------------------------------------


class TestContextRepositorySave:
    @pytest.mark.asyncio
    async def test_save_returns_uuid(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id)

        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        assert isinstance(ctx_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_save_sets_is_current(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id, version=1)

        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        record = await repo.load(ctx_id)
        assert record is not None
        assert record.is_current is True

    @pytest.mark.asyncio
    async def test_save_second_version_marks_first_non_current(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()

        entry1 = _make_vault_entry(session_id, version=1)
        id1 = await repo.save(session_id, entry1)

        entry2 = _make_vault_entry(session_id, version=2)
        id2 = await repo.save(session_id, entry2)
        await db_session.commit()

        r1 = await repo.load(id1)
        r2 = await repo.load(id2)
        assert r1 is not None and r1.is_current is False
        assert r2 is not None and r2.is_current is True


class TestContextRepositoryLoad:
    @pytest.mark.asyncio
    async def test_load_by_id(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id)

        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        record = await repo.load(ctx_id)
        assert record is not None
        assert record.id == ctx_id
        assert record.session_id == session_id
        assert record.version == 1
        assert len(record.encrypted_blob) > 0

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        record = await repo.load(uuid.uuid4())
        assert record is None

    @pytest.mark.asyncio
    async def test_load_current(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()

        entry1 = _make_vault_entry(session_id, version=1)
        await repo.save(session_id, entry1)

        entry2 = _make_vault_entry(session_id, version=2)
        await repo.save(session_id, entry2)
        await db_session.commit()

        current = await repo.load_current(session_id)
        assert current is not None
        assert current.version == 2
        assert current.is_current is True

    @pytest.mark.asyncio
    async def test_load_current_no_versions(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        current = await repo.load_current(uuid.uuid4())
        assert current is None


class TestContextRepositoryList:
    @pytest.mark.asyncio
    async def test_list_versions_ordered_desc(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()

        for v in range(1, 4):
            entry = _make_vault_entry(session_id, version=v)
            await repo.save(session_id, entry)
        await db_session.commit()

        versions = await repo.list_versions(session_id)
        assert len(versions) == 3
        assert versions[0].version == 3
        assert versions[1].version == 2
        assert versions[2].version == 1

    @pytest.mark.asyncio
    async def test_list_versions_pagination(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()

        for v in range(1, 6):
            entry = _make_vault_entry(session_id, version=v)
            await repo.save(session_id, entry)
        await db_session.commit()

        page1 = await repo.list_versions(session_id, limit=2, offset=0)
        page2 = await repo.list_versions(session_id, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].version == 5
        assert page2[0].version == 3

    @pytest.mark.asyncio
    async def test_list_versions_empty(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        versions = await repo.list_versions(uuid.uuid4())
        assert versions == []

    @pytest.mark.asyncio
    async def test_count_versions(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()

        for v in range(1, 4):
            entry = _make_vault_entry(session_id, version=v)
            await repo.save(session_id, entry)
        await db_session.commit()

        count = await repo.count_versions(session_id)
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_versions_zero(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        count = await repo.count_versions(uuid.uuid4())
        assert count == 0


class TestContextRepositoryDelete:
    @pytest.mark.asyncio
    async def test_delete_returns_true(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id)
        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        deleted = await repo.delete(ctx_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        deleted = await repo.delete(uuid.uuid4())
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_removes_from_db(
        self, db_session: AsyncSession
    ) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id)
        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        await repo.delete(ctx_id)
        await db_session.commit()

        record = await repo.load(ctx_id)
        assert record is None


class TestContextRecord:
    @pytest.mark.asyncio
    async def test_record_is_frozen(self, db_session: AsyncSession) -> None:
        repo = ContextRepository(db_session)
        session_id = uuid.uuid4()
        entry = _make_vault_entry(session_id)
        ctx_id = await repo.save(session_id, entry)
        await db_session.commit()

        record = await repo.load(ctx_id)
        assert record is not None
        with pytest.raises(AttributeError):
            record.version = 999  # type: ignore[misc]
