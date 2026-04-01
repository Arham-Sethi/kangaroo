"""API key management endpoints.

Users can create, list, and revoke API keys for SDK/CLI authentication.
Keys are shown in full only once at creation — after that, only the
prefix is visible.

Endpoints:
    POST   /api/v1/api-keys          -- Create a new API key
    GET    /api/v1/api-keys          -- List user's API keys
    DELETE /api/v1/api-keys/{id}     -- Revoke an API key
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.models.db import APIKey, User
from app.core.security.api_key_auth import generate_api_key
from app.core.security.audit import (
    ACTION_API_KEY_CREATE,
    ACTION_API_KEY_REVOKE,
    AuditLogger,
    RESOURCE_API_KEY,
)

router = APIRouter()
logger = structlog.get_logger()

# Maximum API keys per user (prevent abuse)
MAX_KEYS_PER_USER = 25


# -- Request/Response Schemas ------------------------------------------------


class APIKeyCreate(BaseModel):
    """Request to create a new API key."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable name for the key (e.g., 'CLI dev key').",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["contexts:read", "contexts:write", "shifts:execute"],
        description="Permitted scopes for this key.",
    )
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiration timestamp (UTC). None = never expires.",
    )


class APIKeyResponse(BaseModel):
    """API key metadata (no secret)."""

    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    is_active: bool
    created_at: str
    last_used: str | None
    expires_at: str | None


class APIKeyCreatedResponse(APIKeyResponse):
    """Response after creating a key — includes the full key (shown once)."""

    key: str = Field(
        description="The full API key. Store it securely — it will NOT be shown again.",
    )


class APIKeyListResponse(BaseModel):
    """Paginated list of API keys."""

    keys: list[APIKeyResponse]
    total: int


# -- Helpers -----------------------------------------------------------------


def _key_to_response(key: APIKey) -> APIKeyResponse:
    """Convert an APIKey ORM object to a response schema."""
    return APIKeyResponse(
        id=str(key.id),
        name=key.name,
        key_prefix=key.key_prefix,
        scopes=key.scopes if isinstance(key.scopes, list) else [],
        is_active=key.is_active,
        created_at=key.created_at.isoformat() if key.created_at else "",
        last_used=key.last_used.isoformat() if key.last_used else None,
        expires_at=key.expires_at.isoformat() if key.expires_at else None,
    )


# -- Endpoints ---------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=APIKeyCreatedResponse)
async def create_api_key(
    body: APIKeyCreate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIKeyCreatedResponse:
    """Create a new API key.

    The full key is returned only in this response. After creation,
    only the prefix (first 12 chars) is available.
    """
    # Enforce per-user key limit
    count_result = await db.execute(
        select(func.count()).select_from(APIKey).where(
            APIKey.user_id == user.id,
            APIKey.is_active.is_(True),
        )
    )
    active_count = count_result.scalar() or 0
    if active_count >= MAX_KEYS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_KEYS_PER_USER} active API keys allowed. Revoke unused keys first.",
        )

    # Validate scopes
    valid_scopes = {
        "contexts:read", "contexts:write",
        "shifts:execute", "sessions:read", "sessions:write",
        "brain:read", "brain:write",
    }
    invalid = set(body.scopes) - valid_scopes
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scopes: {', '.join(sorted(invalid))}. "
                   f"Valid scopes: {', '.join(sorted(valid_scopes))}.",
        )

    # Validate expiration is in the future
    if body.expires_at is not None:
        now = datetime.now(timezone.utc)
        expires = body.expires_at if body.expires_at.tzinfo else body.expires_at.replace(tzinfo=timezone.utc)
        if expires <= now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expiration must be in the future.",
            )

    # Generate key
    full_key, key_prefix, key_hash = generate_api_key()

    api_key = APIKey(
        user_id=user.id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        name=body.name,
        scopes=body.scopes,
        expires_at=body.expires_at,
    )
    db.add(api_key)

    # Audit log
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_API_KEY_CREATE,
        resource_type=RESOURCE_API_KEY,
        user_id=user.id,
        resource_id=api_key.id,
        ip_address=request.client.host if request.client else None,
        metadata={"name": body.name, "scopes": body.scopes},
    )

    await db.commit()
    await db.refresh(api_key)

    base = _key_to_response(api_key)
    return APIKeyCreatedResponse(
        **base.model_dump(),
        key=full_key,
    )


@router.get("", response_model=APIKeyListResponse)
async def list_api_keys(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(True, description="Only show active (non-revoked) keys."),
) -> APIKeyListResponse:
    """List user's API keys (metadata only, no secrets)."""
    query = select(APIKey).where(APIKey.user_id == user.id)
    if active_only:
        query = query.where(APIKey.is_active.is_(True))
    query = query.order_by(APIKey.created_at.desc())

    result = await db.execute(query)
    keys = result.scalars().all()

    return APIKeyListResponse(
        keys=[_key_to_response(k) for k in keys],
        total=len(keys),
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_200_OK,
    response_model=APIKeyResponse,
)
async def revoke_api_key(
    key_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> APIKeyResponse:
    """Revoke an API key (soft-disable, not delete).

    The key remains in the database for audit purposes but
    can no longer be used for authentication.
    """
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found.",
        )

    if not api_key.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is already revoked.",
        )

    api_key.is_active = False

    # Audit log
    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_API_KEY_REVOKE,
        resource_type=RESOURCE_API_KEY,
        user_id=user.id,
        resource_id=api_key.id,
        ip_address=request.client.host if request.client else None,
        metadata={"name": api_key.name},
    )

    await db.commit()
    await db.refresh(api_key)
    return _key_to_response(api_key)
