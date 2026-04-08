"""Subscription tier definitions — all limits, pricing, and feature gates.

Every resource limit in Kangaroo Shift flows through this module.
Tier configs are frozen dataclasses — immutable, testable, no surprises.

Tiers:
    FREE        — Generous basics after 7-day reverse trial expires
    PRO         — $12/mo (annual) / $16/mo (monthly) — the core revenue tier
    PRO_TEAM    — $9/seat/mo (annual) / $14/seat/mo (monthly) — 5-50 seats
    ENTERPRISE  — Custom pricing, unlimited everything

Design principles:
    1. Every limit is explicit — no hidden caps.
    2. Limits are per-user unless marked per-seat.
    3. Feature gates are boolean — either you have it or you don't.
    4. The reverse trial gives FREE users 7 days of full PRO access.
    5. Usage counters reset on the 1st of each month (UTC).

Usage:
    from app.core.billing.tiers import SubscriptionTier, get_tier_config
    config = get_tier_config(SubscriptionTier.PRO)
    print(config.shifts_per_month)  # -1 means unlimited
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Sentinel for "unlimited" — using -1 is the industry convention
UNLIMITED: int = -1


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""

    FREE = "free"
    PRO = "pro"
    PRO_TEAM = "pro_team"
    ENTERPRISE = "enterprise"


class BillingCycle(str, Enum):
    """Billing frequency."""

    MONTHLY = "monthly"
    ANNUAL = "annual"


@dataclass(frozen=True)
class PricingConfig:
    """Pricing for a tier.

    Attributes:
        monthly_price_usd: Monthly price in USD.
        annual_price_usd: Annual price in USD (per year, not per month).
        per_seat: Whether pricing is per-seat.
        min_seats: Minimum seats for team plans.
        max_seats: Maximum seats for team plans.
    """

    monthly_price_usd: float = 0.0
    annual_price_usd: float = 0.0
    per_seat: bool = False
    min_seats: int = 1
    max_seats: int = 1


@dataclass(frozen=True)
class RateLimitConfig:
    """Per-hour API rate limits.

    Attributes:
        requests_per_hour: Maximum API requests per hour.
        burst_size: Maximum burst above the hourly rate.
    """

    requests_per_hour: int = 30
    burst_size: int = 10


@dataclass(frozen=True)
class TierConfig:
    """Complete configuration for a subscription tier.

    All numeric limits use UNLIMITED (-1) to mean "no cap".
    All boolean flags indicate feature availability.

    Attributes:
        tier: The subscription tier enum.
        display_name: Human-readable tier name.
        pricing: Pricing configuration.
        rate_limit: API rate limiting.

        # Usage limits (per month, per user)
        shifts_per_month: Context shifts allowed per month.
        active_sessions: Maximum concurrent active sessions.
        brain_queries_per_month: Search/recall queries per month.
        tokens_per_month: Total token processing per month.
        retention_days: How long contexts are retained (-1 = forever).
        auto_captures_per_month: Browser extension auto-captures.
        chain_steps_max: Maximum steps in a chain pipeline.
        consensus_models_max: Max models in consensus mode (0 = disabled).
        consensus_lifetime_trials: One-time consensus trials for free users.

        # Team limits
        team_seats_included: Seats included in the plan.
        team_roles_full_rbac: Whether full RBAC is available.

        # Integration limits
        api_keys_max: Maximum API keys.
        webhooks_max: Maximum webhook subscriptions.
        file_upload_mb: Maximum file upload size in MB.

        # Feature gates (boolean)
        cockpit_enabled: Whether the multi-model cockpit is usable.
        cockpit_models_max: Max simultaneous models in cockpit.
        cockpit_view_only: Can see cockpit but not interact.
        smart_dispatch: Whether smart model routing is available.
        digest_daily: Daily digest (vs weekly).
        knowledge_gap_alerts: Whether gap detection alerts are enabled.
        analytics_enabled: Whether usage analytics are available.
        analytics_full: Whether full team analytics are available.
        sso_enabled: SSO/SAML support.

        # Reverse trial
        reverse_trial_days: Days of Pro access on signup (0 = none).

        # Support
        support_level: Support tier (community, email, priority, dedicated).
        support_response_hours: Maximum response time in hours.
    """

    tier: SubscriptionTier
    display_name: str
    pricing: PricingConfig
    rate_limit: RateLimitConfig

    # Usage limits
    shifts_per_month: int = 10
    active_sessions: int = 5
    brain_queries_per_month: int = 15
    tokens_per_month: int = 1_000_000
    retention_days: int = 30
    auto_captures_per_month: int = 0
    chain_steps_max: int = 0
    consensus_models_max: int = 0
    consensus_lifetime_trials: int = 0

    # Team
    team_seats_included: int = 0
    team_roles_full_rbac: bool = False

    # Integrations
    api_keys_max: int = 0
    webhooks_max: int = 0
    file_upload_mb: int = 5

    # Feature gates
    cockpit_enabled: bool = False
    cockpit_models_max: int = 0
    cockpit_view_only: bool = False
    smart_dispatch: bool = False
    digest_daily: bool = False
    knowledge_gap_alerts: bool = False
    analytics_enabled: bool = False
    analytics_full: bool = False
    sso_enabled: bool = False

    # Reverse trial
    reverse_trial_days: int = 0

    # Support
    support_level: str = "community"
    support_response_hours: int = 0


# ── Tier Definitions ─────────────────────────────────────────────────────────


FREE_CONFIG = TierConfig(
    tier=SubscriptionTier.FREE,
    display_name="Free",
    pricing=PricingConfig(),
    rate_limit=RateLimitConfig(requests_per_hour=30, burst_size=10),

    # Usage — generous enough to hook, limited enough to convert
    shifts_per_month=10,
    active_sessions=5,
    brain_queries_per_month=15,
    tokens_per_month=1_000_000,
    retention_days=30,
    auto_captures_per_month=0,  # Manual paste only after trial
    chain_steps_max=0,
    consensus_models_max=0,
    consensus_lifetime_trials=1,  # One free consensus trial ever

    # Team
    team_seats_included=0,
    team_roles_full_rbac=False,

    # Integrations
    api_keys_max=0,
    webhooks_max=0,
    file_upload_mb=5,

    # Features
    cockpit_enabled=False,
    cockpit_models_max=0,
    cockpit_view_only=True,  # Can see but not use
    smart_dispatch=False,
    digest_daily=False,  # Weekly only
    knowledge_gap_alerts=False,
    analytics_enabled=False,
    analytics_full=False,
    sso_enabled=False,

    # 7-day reverse trial on signup
    reverse_trial_days=7,

    # Support
    support_level="community",
    support_response_hours=0,
)


PRO_CONFIG = TierConfig(
    tier=SubscriptionTier.PRO,
    display_name="Pro",
    pricing=PricingConfig(
        monthly_price_usd=16.0,
        annual_price_usd=144.0,  # $12/mo
    ),
    rate_limit=RateLimitConfig(requests_per_hour=600, burst_size=50),

    # Usage — unlimited core, capped extras
    shifts_per_month=UNLIMITED,
    active_sessions=UNLIMITED,
    brain_queries_per_month=UNLIMITED,
    tokens_per_month=5_000_000,
    retention_days=180,
    auto_captures_per_month=UNLIMITED,
    chain_steps_max=5,
    consensus_models_max=0,  # Reserved for Pro Team
    consensus_lifetime_trials=0,

    # Team — 4 seats included (owner + 3 members)
    team_seats_included=4,
    team_roles_full_rbac=False,  # Admin + member only

    # Integrations
    api_keys_max=2,
    webhooks_max=3,
    file_upload_mb=25,

    # Features
    cockpit_enabled=True,
    cockpit_models_max=2,
    cockpit_view_only=False,
    smart_dispatch=True,
    digest_daily=True,
    knowledge_gap_alerts=True,
    analytics_enabled=True,
    analytics_full=False,
    sso_enabled=False,

    # No trial needed — they're paying
    reverse_trial_days=0,

    # Support
    support_level="email",
    support_response_hours=48,
)


PRO_TEAM_CONFIG = TierConfig(
    tier=SubscriptionTier.PRO_TEAM,
    display_name="Pro Team",
    pricing=PricingConfig(
        monthly_price_usd=14.0,
        annual_price_usd=108.0,  # $9/seat/mo
        per_seat=True,
        min_seats=5,
        max_seats=50,
    ),
    rate_limit=RateLimitConfig(requests_per_hour=2000, burst_size=200),

    # Usage — generous per-seat
    shifts_per_month=UNLIMITED,
    active_sessions=UNLIMITED,
    brain_queries_per_month=UNLIMITED,
    tokens_per_month=10_000_000,  # Per seat
    retention_days=UNLIMITED,
    auto_captures_per_month=UNLIMITED,
    chain_steps_max=10,
    consensus_models_max=4,
    consensus_lifetime_trials=0,

    # Team — full features
    team_seats_included=5,  # Minimum purchase
    team_roles_full_rbac=True,

    # Integrations
    api_keys_max=5,  # Per user
    webhooks_max=10,
    file_upload_mb=100,

    # Features
    cockpit_enabled=True,
    cockpit_models_max=4,
    cockpit_view_only=False,
    smart_dispatch=True,
    digest_daily=True,
    knowledge_gap_alerts=True,
    analytics_enabled=True,
    analytics_full=True,
    sso_enabled=True,

    reverse_trial_days=0,

    support_level="priority",
    support_response_hours=24,
)


ENTERPRISE_CONFIG = TierConfig(
    tier=SubscriptionTier.ENTERPRISE,
    display_name="Enterprise",
    pricing=PricingConfig(
        per_seat=True,
        min_seats=50,
        max_seats=UNLIMITED,
    ),
    rate_limit=RateLimitConfig(requests_per_hour=UNLIMITED, burst_size=UNLIMITED),

    # Everything unlimited
    shifts_per_month=UNLIMITED,
    active_sessions=UNLIMITED,
    brain_queries_per_month=UNLIMITED,
    tokens_per_month=UNLIMITED,
    retention_days=UNLIMITED,
    auto_captures_per_month=UNLIMITED,
    chain_steps_max=UNLIMITED,
    consensus_models_max=UNLIMITED,
    consensus_lifetime_trials=0,

    team_seats_included=UNLIMITED,
    team_roles_full_rbac=True,

    api_keys_max=25,
    webhooks_max=UNLIMITED,
    file_upload_mb=500,

    cockpit_enabled=True,
    cockpit_models_max=UNLIMITED,
    cockpit_view_only=False,
    smart_dispatch=True,
    digest_daily=True,
    knowledge_gap_alerts=True,
    analytics_enabled=True,
    analytics_full=True,
    sso_enabled=True,

    reverse_trial_days=0,

    support_level="dedicated",
    support_response_hours=4,
)


# ── Registry ─────────────────────────────────────────────────────────────────

_TIER_REGISTRY: dict[SubscriptionTier, TierConfig] = {
    SubscriptionTier.FREE: FREE_CONFIG,
    SubscriptionTier.PRO: PRO_CONFIG,
    SubscriptionTier.PRO_TEAM: PRO_TEAM_CONFIG,
    SubscriptionTier.ENTERPRISE: ENTERPRISE_CONFIG,
}


def get_tier_config(tier: SubscriptionTier) -> TierConfig:
    """Get the configuration for a subscription tier.

    Args:
        tier: The subscription tier.

    Returns:
        TierConfig with all limits and features.

    Raises:
        ValueError: If the tier is not recognized.
    """
    config = _TIER_REGISTRY.get(tier)
    if config is None:
        raise ValueError(f"Unknown tier: {tier}")
    return config


def get_all_tiers() -> list[TierConfig]:
    """Get all tier configs in display order."""
    return [
        FREE_CONFIG,
        PRO_CONFIG,
        PRO_TEAM_CONFIG,
        ENTERPRISE_CONFIG,
    ]


def is_within_limit(current: int, limit: int) -> bool:
    """Check if current usage is within the limit.

    UNLIMITED (-1) always returns True.
    """
    if limit == UNLIMITED:
        return True
    return current < limit


def remaining_quota(current: int, limit: int) -> int:
    """Get remaining quota.

    Returns UNLIMITED (-1) if the limit is unlimited.
    """
    if limit == UNLIMITED:
        return UNLIMITED
    return max(0, limit - current)
