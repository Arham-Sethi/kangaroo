"""Webhook management endpoints.

Users can create, list, update, delete, and test webhook subscriptions.
Webhooks deliver HMAC-signed HTTP POST payloads for configured events.

Endpoints:
    POST   /api/v1/webhooks          -- Create a webhook
    GET    /api/v1/webhooks          -- List user's webhooks
    PATCH  /api/v1/webhooks/{id}     -- Update webhook (URL, events)
    DELETE /api/v1/webhooks/{id}     -- Delete a webhook
    POST   /api/v1/webhooks/{id}/test -- Send a test delivery
"""

from __future__ import annotations

import secrets
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.core.events.webhook_dispatcher import VALID_EVENTS, WebhookDispatcher
from app.core.models.db import User, Webhook
from app.core.security.audit import (
    ACTION_WEBHOOK_CREATE,
    ACTION_WEBHOOK_DELETE,
    AuditLogger,
    RESOURCE_WEBHOOK,
)

router = APIRouter()
logger = structlog.get_logger()

MAX_WEBHOOKS_PER_USER = 10


# -- Request/Response Schemas ------------------------------------------------


class WebhookCreate(BaseModel):
    url: str = Field(..., max_length=2048, description="URL to deliver events to.")
    events: list[str] = Field(
        ..., min_length=1, description="Event types to subscribe to.",
    )


class WebhookUpdate(BaseModel):
    url: str | None = Field(None, max_length=2048)
    events: list[str] | None = None
    is_active: bool | None = None


class WebhookResponse(BaseModel):
    id: str
    url: str
    events: list[str]
    secret: str
    is_active: bool
    failure_count: int
    last_triggered: str | None
    last_status_code: int | None
    created_at: str


class WebhookListResponse(BaseModel):
    webhooks: list[WebhookResponse]
    total: int


class WebhookTestResponse(BaseModel):
    success: bool
    status_code: int | None
    error: str | None


# -- Helpers -----------------------------------------------------------------


def _webhook_to_response(w: Webhook) -> WebhookResponse:
    return WebhookResponse(
        id=str(w.id),
        url=w.url,
        events=w.events if isinstance(w.events, list) else [],
        secret=w.secret[:8] + "..." if w.secret else "",
        is_active=w.is_active,
        failure_count=w.failure_count,
        last_triggered=w.last_triggered.isoformat() if w.last_triggered else None,
        last_status_code=w.last_status_code,
        created_at=w.created_at.isoformat() if w.created_at else "",
    )


def _validate_events(events: list[str]) -> None:
    invalid = set(events) - VALID_EVENTS
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid events: {', '.join(sorted(invalid))}. "
                   f"Valid: {', '.join(sorted(VALID_EVENTS))}.",
        )


# -- Endpoints ---------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED, response_model=WebhookResponse)
async def create_webhook(
    body: WebhookCreate,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    """Create a new webhook subscription."""
    _validate_events(body.events)

    # Enforce limit
    result = await db.execute(
        select(Webhook).where(Webhook.user_id == user.id)
    )
    existing = result.scalars().all()
    if len(existing) >= MAX_WEBHOOKS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_WEBHOOKS_PER_USER} webhooks allowed.",
        )

    # Check for duplicate URL
    for w in existing:
        if w.url == body.url:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A webhook with this URL already exists.",
            )

    webhook = Webhook(
        user_id=user.id,
        url=body.url,
        events=body.events,
        secret=secrets.token_hex(32),
    )
    db.add(webhook)

    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_WEBHOOK_CREATE,
        resource_type=RESOURCE_WEBHOOK,
        user_id=user.id,
        resource_id=webhook.id,
        ip_address=request.client.host if request.client else None,
        metadata={"url": body.url, "events": body.events},
    )

    await db.commit()
    await db.refresh(webhook)
    return _webhook_to_response(webhook)


@router.get("", response_model=WebhookListResponse)
async def list_webhooks(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WebhookListResponse:
    """List user's webhook subscriptions."""
    result = await db.execute(
        select(Webhook)
        .where(Webhook.user_id == user.id)
        .order_by(Webhook.created_at.desc())
    )
    webhooks = result.scalars().all()
    return WebhookListResponse(
        webhooks=[_webhook_to_response(w) for w in webhooks],
        total=len(webhooks),
    )


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: uuid.UUID,
    body: WebhookUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    """Update a webhook's URL, events, or active status."""
    result = await db.execute(
        select(Webhook).where(
            Webhook.id == webhook_id,
            Webhook.user_id == user.id,
        )
    )
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found.")

    if body.url is not None:
        webhook.url = body.url
    if body.events is not None:
        _validate_events(body.events)
        webhook.events = body.events
    if body.is_active is not None:
        webhook.is_active = body.is_active
        if body.is_active:
            webhook.failure_count = 0  # Reset failures on re-enable

    await db.commit()
    await db.refresh(webhook)
    return _webhook_to_response(webhook)


@router.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_webhook(
    webhook_id: uuid.UUID,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a webhook subscription (hard delete)."""
    result = await db.execute(
        select(Webhook).where(
            Webhook.id == webhook_id,
            Webhook.user_id == user.id,
        )
    )
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found.")

    audit = AuditLogger(db)
    await audit.log(
        action=ACTION_WEBHOOK_DELETE,
        resource_type=RESOURCE_WEBHOOK,
        user_id=user.id,
        resource_id=webhook.id,
        ip_address=request.client.host if request.client else None,
        metadata={"url": webhook.url},
    )

    await db.delete(webhook)
    await db.commit()


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WebhookTestResponse:
    """Send a test delivery to verify webhook is reachable."""
    result = await db.execute(
        select(Webhook).where(
            Webhook.id == webhook_id,
            Webhook.user_id == user.id,
        )
    )
    webhook = result.scalar_one_or_none()
    if webhook is None:
        raise HTTPException(status_code=404, detail="Webhook not found.")

    dispatcher = WebhookDispatcher(db)
    test_payload = {"test": True, "message": "Webhook connectivity test"}

    delivery_result = await dispatcher._deliver(
        webhook, "session.created", test_payload,
    )

    return WebhookTestResponse(
        success=delivery_result.success,
        status_code=delivery_result.status_code,
        error=delivery_result.error,
    )
