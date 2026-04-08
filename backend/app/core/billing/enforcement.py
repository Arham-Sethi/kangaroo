"""Tier enforcement — FastAPI dependencies for permission checks.

Provides reusable FastAPI dependencies that validate whether a user's
current tier allows a specific action. Returns clean 402/429 errors
with upgrade CTAs when limits are exceeded.

Usage in endpoints:
    @router.post("/shifts")
    async def create_shift(
        user: User = Depends(get_current_user),
        _check: None = Depends(require_shifts),
    ):
        ...  # Only reached if the user has shifts remaining

    @router.post("/cockpit/dispatch")
    async def dispatch(
        user: User = Depends(get_current_user),
        _check: None = Depends(require_feature("cockpit_enabled")),
    ):
        ...  # Only reached if tier has cockpit enabled
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status

from app.core.billing.limiter import RateLimiter, RateLimitResult
from app.core.billing.tiers import (
    UNLIMITED,
    SubscriptionTier,
    TierConfig,
    get_tier_config,
    is_within_limit,
)
from app.core.billing.trial import TrialManager
from app.core.billing.usage import UsageMetric, UsageTracker


# Module-level singletons (overrideable in tests)
_usage_tracker = UsageTracker()
_rate_limiter = RateLimiter()
_trial_manager = TrialManager()


def get_usage_tracker() -> UsageTracker:
    return _usage_tracker


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter


def get_trial_manager() -> TrialManager:
    return _trial_manager


class PaywallError(HTTPException):
    """Raised when a user hits a tier limit.

    Returns 402 Payment Required with upgrade information.
    """

    def __init__(
        self,
        resource: str,
        current: int,
        limit: int,
        current_tier: str,
        message: str = "",
    ) -> None:
        detail = {
            "error": "tier_limit_exceeded",
            "resource": resource,
            "current_usage": current,
            "limit": limit,
            "current_tier": current_tier,
            "message": message or f"You've reached your {resource} limit. Upgrade for more.",
            "upgrade_url": "/pricing",
        }
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=detail,
        )


class RateLimitError(HTTPException):
    """Raised when a user exceeds their API rate limit.

    Returns 429 Too Many Requests with retry information.
    """

    def __init__(self, result: RateLimitResult) -> None:
        detail = {
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            **result.to_dict(),
        }
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=result.to_headers(),
        )


class FeatureGateError(HTTPException):
    """Raised when a user tries to use a feature not in their tier."""

    def __init__(
        self,
        feature: str,
        current_tier: str,
        required_tier: str = "Pro",
    ) -> None:
        detail = {
            "error": "feature_not_available",
            "feature": feature,
            "current_tier": current_tier,
            "required_tier": required_tier,
            "message": f"{feature} is available on the {required_tier} plan.",
            "upgrade_url": "/pricing",
        }
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=detail,
        )


def resolve_user_tier(
    user_id: str,
    stored_tier: SubscriptionTier = SubscriptionTier.FREE,
) -> tuple[SubscriptionTier, TierConfig]:
    """Resolve the effective tier for a user (considering trial).

    Returns (effective_tier, tier_config).
    """
    effective = _trial_manager.get_effective_tier(user_id, stored_tier)
    config = get_tier_config(effective)
    return effective, config


def check_usage_limit(
    user_id: str,
    metric: UsageMetric,
    limit: int,
    tier_name: str,
) -> None:
    """Check and raise PaywallError if limit exceeded."""
    result = _usage_tracker.check_within_limit(user_id, metric, limit)
    if not result.allowed:
        raise PaywallError(
            resource=metric.value,
            current=result.current,
            limit=result.limit,
            current_tier=tier_name,
            message=result.message,
        )


def check_rate_limit(user_id: str, config: TierConfig) -> None:
    """Check rate limit and raise RateLimitError if exceeded."""
    result = _rate_limiter.check(user_id, config.rate_limit)
    if not result.allowed:
        raise RateLimitError(result)


def check_feature(feature_name: str, config: TierConfig, tier_name: str) -> None:
    """Check if a feature is enabled and raise FeatureGateError if not."""
    enabled = getattr(config, feature_name, False)
    if not enabled:
        # Determine minimum required tier
        required = "Pro"
        for check_tier in [SubscriptionTier.PRO, SubscriptionTier.PRO_TEAM]:
            check_config = get_tier_config(check_tier)
            if getattr(check_config, feature_name, False):
                required = check_config.display_name
                break

        raise FeatureGateError(
            feature=feature_name,
            current_tier=tier_name,
            required_tier=required,
        )


def record_usage(user_id: str, metric: UsageMetric, amount: int = 1) -> int:
    """Record usage after a successful action.

    Returns the new total.
    """
    return _usage_tracker.increment(user_id, metric, amount)


# ── Reusable Dependency Factories ────────────────────────────────────────────


def require_shifts(user_id: str, tier: SubscriptionTier) -> None:
    """Enforce shift limit for a user."""
    _, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)
    check_usage_limit(
        user_id, UsageMetric.SHIFTS, config.shifts_per_month, config.display_name
    )


def require_brain_query(user_id: str, tier: SubscriptionTier) -> None:
    """Enforce brain query limit."""
    _, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)
    check_usage_limit(
        user_id, UsageMetric.BRAIN_QUERIES, config.brain_queries_per_month, config.display_name
    )


def require_cockpit(user_id: str, tier: SubscriptionTier) -> None:
    """Enforce cockpit access."""
    effective, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)
    check_feature("cockpit_enabled", config, config.display_name)


def require_consensus(user_id: str, tier: SubscriptionTier) -> None:
    """Enforce consensus mode access (Pro Team+ or 1 lifetime trial)."""
    effective, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)

    if config.consensus_models_max > 0 or config.consensus_models_max == UNLIMITED:
        return  # Tier has consensus

    # Check lifetime trial for free users
    if config.consensus_lifetime_trials > 0:
        used = _usage_tracker.get_usage(user_id, UsageMetric.CONSENSUS_TRIALS)
        if used < config.consensus_lifetime_trials:
            return  # Trial available

    raise FeatureGateError(
        feature="Consensus Mode",
        current_tier=config.display_name,
        required_tier="Pro Team",
    )


def require_smart_dispatch(user_id: str, tier: SubscriptionTier) -> None:
    """Enforce smart dispatch access."""
    _, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)
    check_feature("smart_dispatch", config, config.display_name)


def require_chain(user_id: str, tier: SubscriptionTier) -> int:
    """Enforce chain access and return max steps allowed."""
    _, config = resolve_user_tier(user_id, tier)
    check_rate_limit(user_id, config)

    if config.chain_steps_max == 0:
        raise FeatureGateError(
            feature="Chain Pipeline",
            current_tier=config.display_name,
            required_tier="Pro",
        )

    return config.chain_steps_max
