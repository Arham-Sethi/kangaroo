"""Billing and subscription API endpoints.

GET  /api/v1/billing/tiers                -- List all available tiers with pricing
GET  /api/v1/billing/current              -- Get current user's subscription + usage
GET  /api/v1/billing/usage                -- Get detailed usage breakdown
POST /api/v1/billing/trial/start          -- Start reverse trial (auto on signup)
GET  /api/v1/billing/trial/status         -- Check trial status
GET  /api/v1/billing/limits               -- Get current tier limits and remaining quotas
POST /api/v1/billing/checkout             -- Create Stripe checkout session
POST /api/v1/billing/portal               -- Create Stripe billing portal session
POST /api/v1/billing/webhook              -- Stripe webhook handler
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.api.v1.auth import get_current_user
from app.core.billing.enforcement import (
    get_trial_manager,
    get_usage_tracker,
    resolve_user_tier,
)
from app.core.billing.tiers import (
    UNLIMITED,
    SubscriptionTier,
    get_all_tiers,
    get_tier_config,
    remaining_quota,
)
from app.core.billing.usage import UsageMetric
from app.core.models.db import User

router = APIRouter()
logger = structlog.get_logger()

# ── Stripe Price IDs (from environment) ─────────────────────────────────────
# Set these in your .env:
#   STRIPE_SECRET_KEY=sk_...
#   STRIPE_WEBHOOK_SECRET=whsec_...
#   STRIPE_PRICE_PRO_MONTHLY=price_...
#   STRIPE_PRICE_PRO_ANNUAL=price_...
#   STRIPE_PRICE_TEAM_MONTHLY=price_...
#   STRIPE_PRICE_TEAM_ANNUAL=price_...

STRIPE_PRICE_MAP: dict[str, dict[str, str]] = {
    "pro": {
        "monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY", ""),
        "annual": os.getenv("STRIPE_PRICE_PRO_ANNUAL", ""),
    },
    "pro_team": {
        "monthly": os.getenv("STRIPE_PRICE_TEAM_MONTHLY", ""),
        "annual": os.getenv("STRIPE_PRICE_TEAM_ANNUAL", ""),
    },
}


# -- Schemas -----------------------------------------------------------------


class TierPricingResponse(BaseModel):
    """Pricing information for a tier."""

    monthly_price_usd: float
    annual_price_usd: float
    annual_monthly_equivalent: float = 0.0
    per_seat: bool
    min_seats: int
    max_seats: int


class TierResponse(BaseModel):
    """Public tier information."""

    tier: str
    display_name: str
    pricing: TierPricingResponse
    shifts_per_month: int
    active_sessions: int
    brain_queries_per_month: int
    tokens_per_month: int
    retention_days: int
    cockpit_models_max: int
    chain_steps_max: int
    consensus_models_max: int
    team_seats_included: int
    api_keys_max: int
    webhooks_max: int
    file_upload_mb: int
    smart_dispatch: bool
    digest_daily: bool
    knowledge_gap_alerts: bool
    sso_enabled: bool
    support_level: str


class AllTiersResponse(BaseModel):
    """All available tiers."""

    tiers: list[TierResponse]


class UsageMetricResponse(BaseModel):
    """Usage for a single metric."""

    metric: str
    current: int
    limit: int
    remaining: int
    is_unlimited: bool


class UsageResponse(BaseModel):
    """Complete usage breakdown."""

    tier: str
    display_name: str
    period: str
    metrics: list[UsageMetricResponse]


class SubscriptionResponse(BaseModel):
    """Current subscription status."""

    tier: str
    effective_tier: str
    display_name: str
    is_trial_active: bool
    trial_days_remaining: float
    trial_expired: bool


class TrialStatusResponse(BaseModel):
    """Reverse trial status."""

    is_active: bool
    has_been_offered: bool
    days_remaining: float
    trial_tier: str


class LimitsResponse(BaseModel):
    """Current tier limits with remaining quotas."""

    tier: str
    display_name: str
    limits: dict[str, Any]


# -- Endpoints ---------------------------------------------------------------


@router.get("/tiers", response_model=AllTiersResponse)
async def list_tiers() -> AllTiersResponse:
    """List all available subscription tiers with pricing.

    Public endpoint — no authentication required.
    """
    tiers = []
    for config in get_all_tiers():
        annual_monthly = 0.0
        if config.pricing.annual_price_usd > 0:
            annual_monthly = round(config.pricing.annual_price_usd / 12, 2)

        tiers.append(
            TierResponse(
                tier=config.tier.value,
                display_name=config.display_name,
                pricing=TierPricingResponse(
                    monthly_price_usd=config.pricing.monthly_price_usd,
                    annual_price_usd=config.pricing.annual_price_usd,
                    annual_monthly_equivalent=annual_monthly,
                    per_seat=config.pricing.per_seat,
                    min_seats=config.pricing.min_seats,
                    max_seats=config.pricing.max_seats,
                ),
                shifts_per_month=config.shifts_per_month,
                active_sessions=config.active_sessions,
                brain_queries_per_month=config.brain_queries_per_month,
                tokens_per_month=config.tokens_per_month,
                retention_days=config.retention_days,
                cockpit_models_max=config.cockpit_models_max,
                chain_steps_max=config.chain_steps_max,
                consensus_models_max=config.consensus_models_max,
                team_seats_included=config.team_seats_included,
                api_keys_max=config.api_keys_max,
                webhooks_max=config.webhooks_max,
                file_upload_mb=config.file_upload_mb,
                smart_dispatch=config.smart_dispatch,
                digest_daily=config.digest_daily,
                knowledge_gap_alerts=config.knowledge_gap_alerts,
                sso_enabled=config.sso_enabled,
                support_level=config.support_level,
            )
        )

    return AllTiersResponse(tiers=tiers)


@router.get("/current", response_model=SubscriptionResponse)
async def get_current_subscription(
    user: User = Depends(get_current_user),
) -> SubscriptionResponse:
    """Get the current user's subscription status including trial state."""
    user_id = str(user.id)
    stored_tier = SubscriptionTier(user.subscription_tier)

    trial_mgr = get_trial_manager()
    trial_status = trial_mgr.get_trial_status(user_id)

    effective_tier, config = resolve_user_tier(user_id, stored_tier)

    return SubscriptionResponse(
        tier=stored_tier.value,
        effective_tier=effective_tier.value,
        display_name=config.display_name,
        is_trial_active=trial_status.is_active,
        trial_days_remaining=trial_status.days_remaining,
        trial_expired=trial_status.has_been_offered and not trial_status.is_active,
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    user: User = Depends(get_current_user),
) -> UsageResponse:
    """Get detailed usage breakdown for the current billing period."""
    user_id = str(user.id)
    stored_tier = SubscriptionTier(user.subscription_tier)
    effective_tier, config = resolve_user_tier(user_id, stored_tier)

    tracker = get_usage_tracker()
    snapshot = tracker.get_snapshot(user_id)

    # Build metrics list
    metric_mapping = [
        (UsageMetric.SHIFTS, config.shifts_per_month),
        (UsageMetric.BRAIN_QUERIES, config.brain_queries_per_month),
        (UsageMetric.TOKENS_PROCESSED, config.tokens_per_month),
        (UsageMetric.AUTO_CAPTURES, config.auto_captures_per_month),
    ]

    metrics = []
    for metric, limit in metric_mapping:
        current = snapshot.metrics.get(metric, 0)
        rem = remaining_quota(current, limit)
        metrics.append(
            UsageMetricResponse(
                metric=metric.value,
                current=current,
                limit=limit,
                remaining=rem,
                is_unlimited=limit == UNLIMITED,
            )
        )

    # Add lifetime metrics
    consensus_used = snapshot.lifetime_metrics.get(UsageMetric.CONSENSUS_TRIALS, 0)
    metrics.append(
        UsageMetricResponse(
            metric=UsageMetric.CONSENSUS_TRIALS.value,
            current=consensus_used,
            limit=config.consensus_lifetime_trials,
            remaining=remaining_quota(consensus_used, config.consensus_lifetime_trials),
            is_unlimited=False,
        )
    )

    return UsageResponse(
        tier=effective_tier.value,
        display_name=config.display_name,
        period=snapshot.period,
        metrics=metrics,
    )


@router.post("/trial/start", response_model=TrialStatusResponse)
async def start_trial(
    user: User = Depends(get_current_user),
) -> TrialStatusResponse:
    """Start the 7-day reverse trial.

    Automatically called on signup. Can only be started once.
    """
    user_id = str(user.id)
    trial_mgr = get_trial_manager()
    status = trial_mgr.start_trial(user_id)

    return TrialStatusResponse(
        is_active=status.is_active,
        has_been_offered=status.has_been_offered,
        days_remaining=status.days_remaining,
        trial_tier=status.trial_tier.value,
    )


@router.get("/trial/status", response_model=TrialStatusResponse)
async def get_trial_status(
    user: User = Depends(get_current_user),
) -> TrialStatusResponse:
    """Check the current trial status."""
    user_id = str(user.id)
    trial_mgr = get_trial_manager()
    status = trial_mgr.get_trial_status(user_id)

    return TrialStatusResponse(
        is_active=status.is_active,
        has_been_offered=status.has_been_offered,
        days_remaining=status.days_remaining,
        trial_tier=status.trial_tier.value,
    )


@router.get("/limits", response_model=LimitsResponse)
async def get_limits(
    user: User = Depends(get_current_user),
) -> LimitsResponse:
    """Get current tier limits and feature availability."""
    user_id = str(user.id)
    stored_tier = SubscriptionTier(user.subscription_tier)
    effective_tier, config = resolve_user_tier(user_id, stored_tier)

    def _fmt(val: int) -> str | int:
        return "unlimited" if val == UNLIMITED else val

    limits = {
        "shifts_per_month": _fmt(config.shifts_per_month),
        "active_sessions": _fmt(config.active_sessions),
        "brain_queries_per_month": _fmt(config.brain_queries_per_month),
        "tokens_per_month": _fmt(config.tokens_per_month),
        "retention_days": _fmt(config.retention_days),
        "auto_captures_per_month": _fmt(config.auto_captures_per_month),
        "cockpit_enabled": config.cockpit_enabled,
        "cockpit_models_max": _fmt(config.cockpit_models_max),
        "cockpit_view_only": config.cockpit_view_only,
        "smart_dispatch": config.smart_dispatch,
        "chain_steps_max": _fmt(config.chain_steps_max),
        "consensus_models_max": _fmt(config.consensus_models_max),
        "team_seats_included": _fmt(config.team_seats_included),
        "team_roles_full_rbac": config.team_roles_full_rbac,
        "api_keys_max": _fmt(config.api_keys_max),
        "webhooks_max": _fmt(config.webhooks_max),
        "file_upload_mb": config.file_upload_mb,
        "digest_daily": config.digest_daily,
        "knowledge_gap_alerts": config.knowledge_gap_alerts,
        "analytics_enabled": config.analytics_enabled,
        "analytics_full": config.analytics_full,
        "sso_enabled": config.sso_enabled,
        "support_level": config.support_level,
    }

    return LimitsResponse(
        tier=effective_tier.value,
        display_name=config.display_name,
        limits=limits,
    )


# ── Stripe Checkout & Portal ───────────────────────────────────────────────


class CheckoutRequest(BaseModel):
    """Create a Stripe Checkout session."""

    tier: str = Field(..., description="Target tier: 'pro' or 'pro_team'.")
    billing_cycle: str = Field(
        "monthly", description="'monthly' or 'annual'."
    )
    success_url: str = Field(
        default="", description="Redirect URL after successful payment."
    )
    cancel_url: str = Field(
        default="", description="Redirect URL if user cancels."
    )


class CheckoutResponse(BaseModel):
    """Stripe Checkout session URL."""

    checkout_url: str
    session_id: str


class PortalResponse(BaseModel):
    """Stripe Customer Portal URL."""

    portal_url: str


def _get_stripe():
    """Lazy-import stripe to avoid import errors when not installed."""
    try:
        import stripe

        secret_key = os.getenv("STRIPE_SECRET_KEY", "")
        if not secret_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stripe is not configured. Set STRIPE_SECRET_KEY.",
            )
        stripe.api_key = secret_key
        return stripe
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe SDK not installed.",
        )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout_session(
    body: CheckoutRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> CheckoutResponse:
    """Create a Stripe Checkout session for upgrading to a paid plan."""
    stripe = _get_stripe()

    if body.tier not in STRIPE_PRICE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {body.tier}. Must be 'pro' or 'pro_team'.",
        )

    if body.billing_cycle not in ("monthly", "annual"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="billing_cycle must be 'monthly' or 'annual'.",
        )

    price_id = STRIPE_PRICE_MAP[body.tier][body.billing_cycle]
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Stripe price not configured for {body.tier}/{body.billing_cycle}.",
        )

    base_url = str(request.base_url).rstrip("/")
    success_url = body.success_url or f"{base_url}/settings?payment=success"
    cancel_url = body.cancel_url or f"{base_url}/pricing?payment=cancelled"

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            customer_email=user.email,
            client_reference_id=str(user.id),
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "user_id": str(user.id),
                "tier": body.tier,
                "billing_cycle": body.billing_cycle,
            },
        )
        return CheckoutResponse(
            checkout_url=session.url or "",
            session_id=session.id,
        )
    except Exception as e:
        logger.error("stripe_checkout_error", error=str(e), user_id=str(user.id))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create checkout session. Please try again.",
        )


@router.post("/portal", response_model=PortalResponse)
async def create_portal_session(
    request: Request,
    user: User = Depends(get_current_user),
) -> PortalResponse:
    """Create a Stripe Customer Portal session for managing subscriptions."""
    stripe = _get_stripe()

    # Look up or create Stripe customer
    stripe_customer_id = getattr(user, "stripe_customer_id", None)

    if not stripe_customer_id:
        # Find by email
        customers = stripe.Customer.list(email=user.email, limit=1)
        if customers.data:
            stripe_customer_id = customers.data[0].id
        else:
            customer = stripe.Customer.create(
                email=user.email,
                name=getattr(user, "display_name", "") or user.email,
                metadata={"user_id": str(user.id)},
            )
            stripe_customer_id = customer.id

    base_url = str(request.base_url).rstrip("/")

    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{base_url}/settings",
        )
        return PortalResponse(portal_url=session.url)
    except Exception as e:
        logger.error("stripe_portal_error", error=str(e), user_id=str(user.id))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to open billing portal. Please try again.",
        )


@router.post("/webhook")
async def stripe_webhook(request: Request) -> dict:
    """Handle Stripe webhook events (subscription changes, payments).

    This endpoint verifies the webhook signature and processes:
    - checkout.session.completed -> upgrade user tier
    - customer.subscription.updated -> sync tier changes
    - customer.subscription.deleted -> downgrade to free
    - invoice.payment_failed -> flag for follow-up
    """
    stripe = _get_stripe()
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    if webhook_secret:
        try:
            event = stripe.Webhook.construct_event(body, sig, webhook_secret)
        except stripe.error.SignatureVerificationError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook signature.",
            )
    else:
        # Dev mode — no verification
        import json

        event = json.loads(body)

    event_type = event.get("type", "") if isinstance(event, dict) else event.type
    logger.info("stripe_webhook_received", event_type=event_type)

    # Process event types
    if event_type == "checkout.session.completed":
        session_data = event.get("data", {}).get("object", {}) if isinstance(event, dict) else event.data.object
        user_id = (
            session_data.get("metadata", {}).get("user_id")
            if isinstance(session_data, dict)
            else session_data.metadata.get("user_id")
        )
        tier = (
            session_data.get("metadata", {}).get("tier", "pro")
            if isinstance(session_data, dict)
            else session_data.metadata.get("tier", "pro")
        )
        logger.info("checkout_completed", user_id=user_id, tier=tier)
        # TODO: Update user.subscription_tier in DB via async session

    elif event_type == "customer.subscription.deleted":
        logger.info("subscription_cancelled")
        # TODO: Downgrade user to free tier

    elif event_type == "invoice.payment_failed":
        logger.warning("payment_failed")
        # TODO: Send payment failure notification

    return {"status": "ok"}
