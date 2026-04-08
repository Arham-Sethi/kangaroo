"""Context repository — async CRUD for the contexts table.

Encapsulates all database operations for encrypted context blobs.
The repository knows about database rows; it does NOT know about
encryption or UCS deserialization — that's the Vault's job.

Usage:
    repo = ContextRepository(db)
    ctx_id = await repo.save(session_id, entry)
    entry = await repo.load(ctx_id)
    versions = await repo.list_versions(session_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.db import Context
from app.core.storage.vault import VaultEntry


@dataclass(frozen=True)
class ContextRecord:
    """Read-only view of a context row.

    Maps the DB row to a flat, immutable structure
    without exposing the ORM model outside the repository.
    """

    id: uuid.UUID
    session_id: uuid.UUID
    version: int
    encrypted_blob: bytes
    encryption_key_id: str
    blob_size_bytes: int
    compression_ratio: float | None
    parent_version_id: uuid.UUID | None
    is_current: bool
    metadata: dict[str, Any]
    created_at: datetime


class ContextRepository:
    """Async repository for contexts table.

    All methods are pure database operations — no encryption,
    no business logic. Keeps the data layer clean.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save(
        self,
        session_id: uuid.UUID,
        entry: VaultEntry,
        encryption_key_id: str = "default-v1",
    ) -> uuid.UUID:
        """Save an encrypted context blob to the database.

        If this is a new version for an existing session, marks
        previous versions as non-current.

        Args:
            session_id: The session this context belongs to.
            entry: VaultEntry from the Vault (encrypted blob + metadata).
            encryption_key_id: Key identifier for rotation support.

        Returns:
            The context row UUID.
        """
        # Mark previous current version as non-current
        await self._db.execute(
            update(Context)
            .where(
                Context.session_id == session_id,
                Context.is_current.is_(True),
            )
            .values(is_current=False)
        )

        context = Context(
            id=entry.id,
            session_id=session_id,
            version=entry.version,
            encrypted_blob=entry.encrypted_blob,
            encryption_key_id=encryption_key_id,
            blob_size_bytes=entry.blob_size_bytes,
            compression_ratio=entry.compression_ratio,
            parent_version_id=entry.parent_version_id,
            is_current=True,
            extra_data=entry.metadata,
        )
        self._db.add(context)
        await self._db.flush()
        return context.id

    async def load(self, context_id: uuid.UUID) -> ContextRecord | None:
        """Load a context by ID.

        Returns None if not found.
        """
        result = await self._db.execute(
            select(Context).where(Context.id == context_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_record(row)

    async def load_current(self, session_id: uuid.UUID) -> ContextRecord | None:
        """Load the current (latest) context for a session.

        Returns None if the session has no saved contexts.
        """
        result = await self._db.execute(
            select(Context).where(
                Context.session_id == session_id,
                Context.is_current.is_(True),
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_record(row)

    async def list_versions(
        self,
        session_id: uuid.UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ContextRecord]:
        """List all context versions for a session, newest first.

        Returns metadata only (no encrypted blobs) for efficiency.
        """
        result = await self._db.execute(
            select(Context)
            .where(Context.session_id == session_id)
            .order_by(Context.version.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = result.scalars().all()
        return [self._to_record(r) for r in rows]

    async def count_versions(self, session_id: uuid.UUID) -> int:
        """Count total context versions for a session."""
        from sqlalchemy import func

        result = await self._db.execute(
            select(func.count())
            .select_from(Context)
            .where(Context.session_id == session_id)
        )
        return result.scalar() or 0

    async def delete(self, context_id: uuid.UUID) -> bool:
        """Delete a context by ID. Returns True if deleted."""
        result = await self._db.execute(
            select(Context).where(Context.id == context_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return False
        await self._db.delete(row)
        return True

    def _to_record(self, row: Context) -> ContextRecord:
        """Convert ORM model to immutable record."""
        return ContextRecord(
            id=row.id,
            session_id=row.session_id,
            version=row.version,
            encrypted_blob=row.encrypted_blob,
            encryption_key_id=row.encryption_key_id,
            blob_size_bytes=row.blob_size_bytes,
            compression_ratio=row.compression_ratio,
            parent_version_id=row.parent_version_id,
            is_current=row.is_current,
            metadata=row.extra_data if isinstance(row.extra_data, dict) else {},
            created_at=row.created_at if row.created_at else datetime.now(timezone.utc),
        )
