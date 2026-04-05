"""Encrypted blob storage for UCS contexts.

The Vault encrypts UCS data at rest using AES-256-GCM (via EncryptionEngine)
and optionally compresses with zlib before encryption. Session-based AAD
(Associated Authenticated Data) binds each blob to a specific session,
preventing cross-session tampering.

Usage:
    from app.core.storage.vault import Vault

    vault = Vault(master_key="system-secret")
    entry = vault.store(ucs, session_id=session_uuid)
    retrieved_ucs = vault.retrieve(entry)
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.core.models.ucs import UniversalContextSchema
from app.core.security.encryption import EncryptionEngine, EncryptedPayload


@dataclass(frozen=True)
class VaultEntry:
    """Immutable metadata and encrypted blob for a stored UCS.

    Fields:
        id: Unique identifier for this vault entry.
        session_id: Session this context belongs to (used as AAD).
        version: Context version number.
        encrypted_blob: Raw encrypted bytes.
        blob_size_bytes: Size of the encrypted blob.
        original_size_bytes: Size of the uncompressed JSON.
        compression_ratio: original / blob (higher = more compression).
        parent_version_id: Parent version for branching.
        created_at: When this entry was created.
        metadata: Searchable metadata (source_llm, entity_count, etc.).
    """

    id: UUID
    session_id: UUID
    version: int
    encrypted_blob: bytes
    blob_size_bytes: int
    original_size_bytes: int
    compression_ratio: float
    parent_version_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class Vault:
    """Encrypted UCS storage with optional zlib compression.

    Each store() call:
        1. Serializes UCS to JSON
        2. Optionally compresses with zlib
        3. Encrypts with AES-256-GCM (session_id as AAD)

    Each retrieve() call reverses the process.
    """

    def __init__(
        self,
        master_key: str,
        compress: bool = True,
        compression_level: int = 6,
        kdf_iterations: int = 10_000,
    ) -> None:
        """Initialize vault with a master encryption key.

        Args:
            master_key: Secret key for AES-256-GCM encryption.
            compress: Whether to zlib-compress before encrypting.
            compression_level: zlib compression level (1=fast, 9=best).
            kdf_iterations: PBKDF2 iterations (low default for dev speed).
        """
        self._engine = EncryptionEngine(
            master_key=master_key,
            kdf_iterations=kdf_iterations,
        )
        self._compress = compress
        self._compression_level = compression_level

    def store(
        self,
        ucs: UniversalContextSchema,
        *,
        session_id: UUID | None = None,
        version: int = 1,
        parent_version_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VaultEntry:
        """Encrypt and store a UCS.

        Args:
            ucs: The Universal Context Schema to store.
            session_id: Session binding (auto-generated if omitted).
            version: Version number for this context.
            parent_version_id: Parent version for branching history.
            metadata: Additional searchable metadata.

        Returns:
            VaultEntry with encrypted blob and metadata.
        """
        session_id = session_id or uuid4()
        aad = session_id.bytes

        json_bytes = ucs.model_dump_json().encode("utf-8")
        original_size = len(json_bytes)

        if self._compress:
            payload_bytes = zlib.compress(json_bytes, self._compression_level)
        else:
            payload_bytes = json_bytes

        encrypted = self._engine.encrypt(payload_bytes, associated_data=aad)
        blob = encrypted.to_bytes()

        entry_metadata: dict[str, Any] = {
            "source_llm": ucs.session_meta.source_llm.value,
            "message_count": ucs.session_meta.message_count,
            "entity_count": len(ucs.entities),
            "compressed": self._compress,
        }
        if metadata:
            entry_metadata.update(metadata)

        blob_size = len(blob)
        ratio = original_size / blob_size if blob_size > 0 else 0.0

        return VaultEntry(
            id=uuid4(),
            session_id=session_id,
            version=version,
            encrypted_blob=blob,
            blob_size_bytes=blob_size,
            original_size_bytes=original_size,
            compression_ratio=ratio,
            parent_version_id=parent_version_id,
            metadata=entry_metadata,
        )

    def retrieve(self, entry: VaultEntry) -> UniversalContextSchema:
        """Decrypt and deserialize a stored UCS.

        Args:
            entry: VaultEntry from a previous store() call.

        Returns:
            The original UniversalContextSchema.

        Raises:
            ValueError: If decryption fails (wrong key, tampered session_id,
                        corrupted blob).
        """
        aad = entry.session_id.bytes

        try:
            encrypted = EncryptedPayload.from_bytes(entry.encrypted_blob)
        except ValueError as e:
            raise ValueError(f"Failed to parse encrypted blob: {e}") from e

        decrypted = self._engine.decrypt(encrypted, associated_data=aad)

        compressed = entry.metadata.get("compressed", True)
        if compressed:
            try:
                json_bytes = zlib.decompress(decrypted)
            except zlib.error as e:
                raise ValueError(f"Decompression failed: {e}") from e
        else:
            json_bytes = decrypted

        data = json.loads(json_bytes.decode("utf-8"))
        return UniversalContextSchema(**data)
