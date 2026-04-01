"""Tests for encrypted blob storage (Vault).

Tests cover:
    - Store/retrieve round-trip
    - Encryption binding (session-based AAD)
    - Compression (zlib)
    - VaultEntry metadata correctness
    - Error handling (wrong key, corrupted data)
    - Compression disabled mode
"""

from uuid import uuid4

import pytest

from app.core.models.ucs import (
    Entity,
    EntityType,
    SessionMeta,
    SourceLLM,
    UniversalContextSchema,
)
from app.core.storage.vault import Vault, VaultEntry


# -- Helpers -----------------------------------------------------------------


def _minimal_ucs() -> UniversalContextSchema:
    """Create a minimal UCS for vault tests."""
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=3,
            total_tokens=500,
        ),
    )


def _rich_ucs() -> UniversalContextSchema:
    """Create a UCS with entities for richer tests."""
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.ANTHROPIC,
            source_model="claude-3-opus",
            message_count=10,
            total_tokens=5000,
        ),
        entities=(
            Entity(
                name="Python",
                type=EntityType.TECHNOLOGY,
                importance=0.9,
            ),
            Entity(
                name="FastAPI",
                type=EntityType.TECHNOLOGY,
                importance=0.8,
            ),
        ),
    )


# == Store & Retrieve Round-Trip =============================================


class TestVaultRoundTrip:
    """Tests for store -> retrieve cycle."""

    @pytest.fixture
    def vault(self) -> Vault:
        return Vault(master_key="test-vault-key-12345")

    def test_store_returns_vault_entry(self, vault: Vault) -> None:
        ucs = _minimal_ucs()
        entry = vault.store(ucs)
        assert isinstance(entry, VaultEntry)
        assert entry.id is not None
        assert entry.session_id is not None
        assert entry.version == 1
        assert entry.encrypted_blob is not None
        assert len(entry.encrypted_blob) > 0

    def test_retrieve_matches_original(self, vault: Vault) -> None:
        ucs = _minimal_ucs()
        entry = vault.store(ucs)
        retrieved = vault.retrieve(entry)
        assert retrieved.session_meta.source_llm == ucs.session_meta.source_llm
        assert retrieved.session_meta.message_count == ucs.session_meta.message_count

    def test_round_trip_with_entities(self, vault: Vault) -> None:
        ucs = _rich_ucs()
        entry = vault.store(ucs)
        retrieved = vault.retrieve(entry)
        assert len(retrieved.entities) == 2
        assert retrieved.entities[0].name == "Python"
        assert retrieved.entities[1].name == "FastAPI"

    def test_round_trip_preserves_full_ucs(self, vault: Vault) -> None:
        ucs = _rich_ucs()
        entry = vault.store(ucs)
        retrieved = vault.retrieve(entry)
        assert retrieved.model_dump_json() == ucs.model_dump_json()


# == VaultEntry Metadata =====================================================


class TestVaultEntryMetadata:
    """Tests for VaultEntry properties."""

    @pytest.fixture
    def vault(self) -> Vault:
        return Vault(master_key="metadata-test-key")

    def test_blob_size_is_positive(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.blob_size_bytes > 0

    def test_original_size_is_positive(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.original_size_bytes > 0

    def test_compression_ratio_reasonable(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert 0.0 < entry.compression_ratio <= 2.0

    def test_metadata_has_source_llm(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.metadata["source_llm"] == "openai"

    def test_metadata_has_message_count(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.metadata["message_count"] == 3

    def test_metadata_has_entity_count(self, vault: Vault) -> None:
        entry = vault.store(_rich_ucs())
        assert entry.metadata["entity_count"] == 2

    def test_metadata_has_compressed_flag(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.metadata["compressed"] is True

    def test_custom_metadata_included(self, vault: Vault) -> None:
        entry = vault.store(
            _minimal_ucs(),
            metadata={"custom_key": "custom_value"},
        )
        assert entry.metadata["custom_key"] == "custom_value"
        assert "source_llm" in entry.metadata  # standard metadata still present

    def test_session_id_provided(self, vault: Vault) -> None:
        sid = uuid4()
        entry = vault.store(_minimal_ucs(), session_id=sid)
        assert entry.session_id == sid

    def test_session_id_auto_generated(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.session_id is not None

    def test_version_number(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs(), version=5)
        assert entry.version == 5

    def test_parent_version_id(self, vault: Vault) -> None:
        parent = uuid4()
        entry = vault.store(_minimal_ucs(), parent_version_id=parent)
        assert entry.parent_version_id == parent

    def test_created_at_is_set(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        assert entry.created_at is not None

    def test_entry_is_frozen(self, vault: Vault) -> None:
        entry = vault.store(_minimal_ucs())
        with pytest.raises(AttributeError):
            entry.version = 99  # type: ignore[misc]


# == Session Binding (AAD) ===================================================


class TestSessionBinding:
    """Tests for session-based encryption binding."""

    def test_different_sessions_produce_different_blobs(self) -> None:
        vault = Vault(master_key="session-test")
        ucs = _minimal_ucs()
        e1 = vault.store(ucs, session_id=uuid4())
        e2 = vault.store(ucs, session_id=uuid4())
        assert e1.encrypted_blob != e2.encrypted_blob

    def test_wrong_session_id_fails_decryption(self) -> None:
        vault = Vault(master_key="session-test")
        ucs = _minimal_ucs()
        entry = vault.store(ucs, session_id=uuid4())

        # Tamper with session_id (changes AAD)
        tampered = VaultEntry(
            id=entry.id,
            session_id=uuid4(),  # wrong session!
            version=entry.version,
            encrypted_blob=entry.encrypted_blob,
            blob_size_bytes=entry.blob_size_bytes,
            original_size_bytes=entry.original_size_bytes,
            compression_ratio=entry.compression_ratio,
            parent_version_id=entry.parent_version_id,
            created_at=entry.created_at,
            metadata=entry.metadata,
        )
        with pytest.raises(ValueError):
            vault.retrieve(tampered)


# == Compression =============================================================


class TestCompression:
    """Tests for zlib compression behavior."""

    def test_compressed_blob_smaller(self) -> None:
        vault_compressed = Vault(master_key="compress-test", compress=True)
        vault_raw = Vault(master_key="compress-test", compress=False)
        ucs = _rich_ucs()
        e_compressed = vault_compressed.store(ucs)
        e_raw = vault_raw.store(ucs)
        # Compressed should be smaller for typical JSON
        assert e_compressed.blob_size_bytes <= e_raw.blob_size_bytes

    def test_no_compression_mode(self) -> None:
        vault = Vault(master_key="no-compress", compress=False)
        ucs = _minimal_ucs()
        entry = vault.store(ucs)
        assert entry.metadata["compressed"] is False
        retrieved = vault.retrieve(entry)
        assert retrieved.session_meta.source_llm == ucs.session_meta.source_llm

    def test_compression_level(self) -> None:
        vault_fast = Vault(master_key="level-test", compression_level=1)
        vault_best = Vault(master_key="level-test", compression_level=9)
        ucs = _rich_ucs()
        e_fast = vault_fast.store(ucs)
        e_best = vault_best.store(ucs)
        # Both should be retrievable
        assert vault_fast.retrieve(e_fast).model_dump_json() == ucs.model_dump_json()
        assert vault_best.retrieve(e_best).model_dump_json() == ucs.model_dump_json()


# == Error Handling ==========================================================


class TestVaultErrors:
    """Tests for error conditions."""

    def test_wrong_key_fails(self) -> None:
        vault1 = Vault(master_key="key-1")
        vault2 = Vault(master_key="key-2")
        entry = vault1.store(_minimal_ucs())
        with pytest.raises(ValueError):
            vault2.retrieve(entry)

    def test_corrupted_blob_fails(self) -> None:
        vault = Vault(master_key="corrupt-test")
        entry = vault.store(_minimal_ucs())
        corrupted = VaultEntry(
            id=entry.id,
            session_id=entry.session_id,
            version=entry.version,
            encrypted_blob=b"totally-corrupted-data-that-is-long-enough-for-parsing" * 2,
            blob_size_bytes=entry.blob_size_bytes,
            original_size_bytes=entry.original_size_bytes,
            compression_ratio=entry.compression_ratio,
            parent_version_id=entry.parent_version_id,
            created_at=entry.created_at,
            metadata=entry.metadata,
        )
        with pytest.raises(ValueError):
            vault.retrieve(corrupted)
