"""Tests for SessionStore — orchestrates Vault encryption with DB persistence.

Tests cover:
    - Save UCS (encrypt + persist) round-trip
    - Load current version (decrypt from DB)
    - Load specific version by context_id
    - Auto-incrementing version numbers
    - List versions metadata
    - Delete a version
    - SaveResult correctness
    - Load returns None when empty
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.ucs import (
    Entity,
    EntityType,
    SessionMeta,
    SourceLLM,
    UniversalContextSchema,
)
from app.core.storage.session_store import SessionStore, SaveResult, VersionInfo


# -- Helpers -----------------------------------------------------------------

MASTER_KEY = "test-session-store-key-32chars!"


def _minimal_ucs() -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=5,
            total_tokens=800,
        ),
    )


def _rich_ucs() -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.ANTHROPIC,
            source_model="claude-3-opus",
            message_count=10,
            total_tokens=5000,
        ),
        entities=(
            Entity(name="Python", type=EntityType.TECHNOLOGY, importance=0.9),
            Entity(name="FastAPI", type=EntityType.TECHNOLOGY, importance=0.8),
        ),
    )


# -- Tests -------------------------------------------------------------------


class TestSessionStoreSave:
    @pytest.mark.asyncio
    async def test_save_returns_save_result(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        result = await store.save(session_id, _minimal_ucs())
        await db_session.commit()

        assert isinstance(result, SaveResult)
        assert isinstance(result.context_id, uuid.UUID)
        assert result.version == 1
        assert result.blob_size_bytes > 0
        assert result.compression_ratio > 0

    @pytest.mark.asyncio
    async def test_save_auto_increments_version(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        r1 = await store.save(session_id, _minimal_ucs())
        r2 = await store.save(session_id, _rich_ucs())
        r3 = await store.save(session_id, _minimal_ucs())
        await db_session.commit()

        assert r1.version == 1
        assert r2.version == 2
        assert r3.version == 3

    @pytest.mark.asyncio
    async def test_save_with_metadata(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        result = await store.save(
            session_id,
            _minimal_ucs(),
            metadata={"custom_tag": "test"},
        )
        await db_session.commit()

        assert result.version == 1


class TestSessionStoreLoad:
    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        await store.save(session_id, _rich_ucs())
        await db_session.commit()

        loaded = await store.load(session_id)
        assert loaded is not None
        assert loaded.session_meta.source_llm == SourceLLM.ANTHROPIC
        assert loaded.session_meta.source_model == "claude-3-opus"
        assert loaded.session_meta.message_count == 10
        assert len(loaded.entities) == 2

    @pytest.mark.asyncio
    async def test_load_returns_current_version(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        await store.save(session_id, _minimal_ucs())  # v1
        await store.save(session_id, _rich_ucs())      # v2 (current)
        await db_session.commit()

        loaded = await store.load(session_id)
        assert loaded is not None
        # v2 is the rich UCS with Anthropic
        assert loaded.session_meta.source_llm == SourceLLM.ANTHROPIC

    @pytest.mark.asyncio
    async def test_load_specific_version(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        r1 = await store.save(session_id, _minimal_ucs())  # v1 OpenAI
        await store.save(session_id, _rich_ucs())            # v2 Anthropic
        await db_session.commit()

        loaded = await store.load(session_id, context_id=r1.context_id)
        assert loaded is not None
        assert loaded.session_meta.source_llm == SourceLLM.OPENAI

    @pytest.mark.asyncio
    async def test_load_empty_session_returns_none(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        loaded = await store.load(uuid.uuid4())
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_nonexistent_context_id_returns_none(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        loaded = await store.load(uuid.uuid4(), context_id=uuid.uuid4())
        assert loaded is None


class TestSessionStoreListVersions:
    @pytest.mark.asyncio
    async def test_list_versions(self, db_session: AsyncSession) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        await store.save(session_id, _minimal_ucs())
        await store.save(session_id, _rich_ucs())
        await db_session.commit()

        versions = await store.list_versions(session_id)
        assert len(versions) == 2
        assert all(isinstance(v, VersionInfo) for v in versions)
        # Newest first
        assert versions[0].version == 2
        assert versions[1].version == 1

    @pytest.mark.asyncio
    async def test_list_versions_current_flag(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        await store.save(session_id, _minimal_ucs())
        await store.save(session_id, _rich_ucs())
        await db_session.commit()

        versions = await store.list_versions(session_id)
        assert versions[0].is_current is True
        assert versions[1].is_current is False

    @pytest.mark.asyncio
    async def test_list_versions_empty(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        versions = await store.list_versions(uuid.uuid4())
        assert versions == []

    @pytest.mark.asyncio
    async def test_version_info_has_metadata(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        await store.save(session_id, _minimal_ucs())
        await db_session.commit()

        versions = await store.list_versions(session_id)
        v = versions[0]
        assert v.blob_size_bytes > 0
        assert v.context_id is not None
        assert isinstance(v.metadata, dict)


class TestSessionStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_version(self, db_session: AsyncSession) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        session_id = uuid.uuid4()

        result = await store.save(session_id, _minimal_ucs())
        await db_session.commit()

        deleted = await store.delete_version(result.context_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self, db_session: AsyncSession
    ) -> None:
        store = SessionStore(db_session, master_key=MASTER_KEY)
        deleted = await store.delete_version(uuid.uuid4())
        assert deleted is False
