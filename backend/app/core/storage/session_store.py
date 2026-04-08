"""Session store — orchestrates Vault encryption with database persistence.

This is the high-level API for storing and retrieving UCS documents.
It coordinates:
    1. Vault: encrypt/decrypt UCS <-> binary blob
    2. Repository: save/load binary blob <-> database row
    3. VersionGraph: track version history and branching

Usage:
    store = SessionStore(db, master_key="secret")
    version_id = await store.save(session_id, ucs)
    ucs = await store.load(session_id)
    versions = await store.list_versions(session_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.ucs import UniversalContextSchema
from app.core.storage.repository import ContextRecord, ContextRepository
from app.core.storage.vault import Vault, VaultEntry


@dataclass(frozen=True)
class SaveResult:
    """Result of saving a UCS document.

    Attributes:
        context_id: The database row UUID.
        version: The version number.
        blob_size_bytes: Size of encrypted blob.
        compression_ratio: Compression efficiency.
    """

    context_id: uuid.UUID
    version: int
    blob_size_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class VersionInfo:
    """Version metadata without the encrypted blob.

    Used for listing versions without loading full content.
    """

    context_id: uuid.UUID
    version: int
    is_current: bool
    blob_size_bytes: int
    compression_ratio: float | None
    parent_version_id: uuid.UUID | None
    created_at: datetime
    metadata: dict[str, Any]


class SessionStore:
    """High-level store that combines Vault encryption with DB persistence.

    Provides a clean API for save/load/list without exposing
    encryption or database details to callers.
    """

    def __init__(
        self,
        db: AsyncSession,
        master_key: str | bytes,
        encryption_key_id: str = "default-v1",
    ) -> None:
        self._repo = ContextRepository(db)
        self._vault = Vault(master_key=master_key)
        self._encryption_key_id = encryption_key_id
        self._db = db

    async def save(
        self,
        session_id: uuid.UUID,
        ucs: UniversalContextSchema,
        *,
        parent_version_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SaveResult:
        """Encrypt and persist a UCS document.

        Automatically increments the version number based on
        existing versions for this session.

        Args:
            session_id: The session to save under.
            ucs: The UCS document to encrypt and store.
            parent_version_id: For branching — the parent version.
            metadata: Extra searchable metadata.

        Returns:
            SaveResult with context_id, version, and size info.
        """
        # Determine next version number
        version_count = await self._repo.count_versions(session_id)
        next_version = version_count + 1

        # Encrypt via Vault
        entry = self._vault.store(
            ucs,
            session_id=session_id,
            version=next_version,
            parent_version_id=parent_version_id,
            metadata=metadata,
        )

        # Persist to database
        context_id = await self._repo.save(
            session_id=session_id,
            entry=entry,
            encryption_key_id=self._encryption_key_id,
        )

        return SaveResult(
            context_id=context_id,
            version=next_version,
            blob_size_bytes=entry.blob_size_bytes,
            compression_ratio=entry.compression_ratio,
        )

    async def load(
        self,
        session_id: uuid.UUID,
        *,
        context_id: uuid.UUID | None = None,
    ) -> UniversalContextSchema | None:
        """Load and decrypt a UCS document.

        If context_id is given, loads that specific version.
        Otherwise loads the current (latest) version.

        Returns None if no context is found.
        """
        if context_id is not None:
            record = await self._repo.load(context_id)
        else:
            record = await self._repo.load_current(session_id)

        if record is None:
            return None

        # Reconstruct VaultEntry from DB record
        vault_entry = self._record_to_vault_entry(record, session_id)
        return self._vault.retrieve(vault_entry)

    async def list_versions(
        self,
        session_id: uuid.UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[VersionInfo]:
        """List all versions for a session (newest first).

        Returns metadata only — does NOT decrypt content.
        """
        records = await self._repo.list_versions(
            session_id, limit=limit, offset=offset,
        )
        return [
            VersionInfo(
                context_id=r.id,
                version=r.version,
                is_current=r.is_current,
                blob_size_bytes=r.blob_size_bytes,
                compression_ratio=r.compression_ratio,
                parent_version_id=r.parent_version_id,
                created_at=r.created_at,
                metadata=r.metadata,
            )
            for r in records
        ]

    async def delete_version(self, context_id: uuid.UUID) -> bool:
        """Delete a specific context version. Returns True if deleted."""
        return await self._repo.delete(context_id)

    def _record_to_vault_entry(
        self,
        record: ContextRecord,
        session_id: uuid.UUID,
    ) -> VaultEntry:
        """Convert a DB record back to a VaultEntry for decryption."""
        return VaultEntry(
            id=record.id,
            session_id=session_id,
            version=record.version,
            encrypted_blob=record.encrypted_blob,
            blob_size_bytes=record.blob_size_bytes,
            original_size_bytes=0,  # Not stored in DB, not needed for decryption
            compression_ratio=record.compression_ratio or 0.0,
            parent_version_id=record.parent_version_id,
            created_at=record.created_at,
            metadata=record.metadata,
        )
