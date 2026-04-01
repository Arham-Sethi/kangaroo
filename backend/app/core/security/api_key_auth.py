"""API key authentication dependency for SDK/CLI access.

API keys provide an alternative to JWT tokens for programmatic access.
They are:
    - Long-lived (optional expiration)
    - Scoped (fine-grained permissions)
    - Revocable (instant deactivation)
    - Auditable (last_used tracking)

Key format: ks_live_<32 hex chars> (48 chars total)
    - Prefix "ks_live_" is stored in plaintext for identification
    - Full key is bcrypt-hashed (same cost factor as passwords)
    - Only shown once at creation time

Usage:
    from app.core.security.api_key_auth import get_current_user_by_api_key

    @router.get("/data")
    async def get_data(user: User = Depends(get_current_user_by_api_key)):
        ...
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone

import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.models.db import APIKey, User

# API key header scheme
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Key format constants
KEY_PREFIX = "ks_live_"
KEY_RANDOM_BYTES = 32  # 32 hex chars = 128 bits of entropy


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (full_key, key_prefix, key_hash):
            - full_key: the complete key to show the user once
            - key_prefix: first 12 chars for display in UI
            - key_hash: bcrypt hash for storage
    """
    random_part = secrets.token_hex(KEY_RANDOM_BYTES // 2)
    full_key = f"{KEY_PREFIX}{random_part}"
    key_prefix = full_key[:12]
    key_hash = bcrypt.hashpw(
        full_key.encode("utf-8"),
        bcrypt.gensalt(rounds=12),
    ).decode("utf-8")
    return full_key, key_prefix, key_hash


def verify_api_key(raw_key: str, key_hash: str) -> bool:
    """Verify an API key against its bcrypt hash (constant-time)."""
    return bcrypt.checkpw(
        raw_key.encode("utf-8"),
        key_hash.encode("utf-8"),
    )


async def get_current_user_by_api_key(
    api_key: str | None = Security(_api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Authenticate a user via API key in X-API-Key header.

    Validates:
        1. Header is present
        2. Key matches a stored hash
        3. Key is active (not revoked)
        4. Key is not expired
        5. User is active (not deleted)

    Side effect: updates last_used timestamp on successful auth.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Set X-API-Key header.",
        )

    if not api_key.startswith(KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format.",
        )

    # Find candidate keys by prefix (narrows bcrypt checks)
    prefix = api_key[:12]
    result = await db.execute(
        select(APIKey).where(
            APIKey.key_prefix == prefix,
            APIKey.is_active.is_(True),
        )
    )
    candidates = result.scalars().all()

    matched_key: APIKey | None = None
    for candidate in candidates:
        if verify_api_key(api_key, candidate.key_hash):
            matched_key = candidate
            break

    if matched_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    # Check expiration
    now = datetime.now(timezone.utc)
    if matched_key.expires_at is not None and matched_key.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired.",
        )

    # Update last_used
    await db.execute(
        update(APIKey)
        .where(APIKey.id == matched_key.id)
        .values(last_used=now)
    )
    await db.commit()

    # Load the user
    user_result = await db.execute(
        select(User).where(
            User.id == matched_key.user_id,
            User.deleted_at.is_(None),
        )
    )
    user = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
        )

    return user
