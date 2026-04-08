"""Tests for subscription tier definitions and helper functions.

Tests cover:
    - Tier enum values and completeness
    - TierConfig frozen immutability
    - PricingConfig values per tier
    - RateLimitConfig values per tier
    - Feature gates per tier
    - get_tier_config() lookup and error handling
    - get_all_tiers() ordering and completeness
    - is_within_limit() with normal and unlimited values
    - remaining_quota() with normal and unlimited values
    - UNLIMITED sentinel behavior
"""

import pytest

from app.core.billing.tiers import (
    UNLIMITED,
    BillingCycle,
    PricingConfig,
    RateLimitConfig,
    SubscriptionTier,
    TierConfig,
    get_all_tiers,
    get_tier_config,
    is_within_limit,
    remaining_quota,
    FREE_CONFIG,
    PRO_CONFIG,
    PRO_TEAM_CONFIG,
    ENTERPRISE_CONFIG,
)


class TestSubscriptionTier:
    """Test the SubscriptionTier enum."""

    def test_all_tiers_exist(self) -> None:
        assert SubscriptionTier.FREE == "free"
        assert SubscriptionTier.PRO == "pro"
        assert SubscriptionTier.PRO_TEAM == "pro_team"
        assert SubscriptionTier.ENTERPRISE == "enterprise"

    def test_tier_count(self) -> None:
        assert len(SubscriptionTier) == 4

    def test_tier_is_str_enum(self) -> None:
        assert isinstance(SubscriptionTier.FREE, str)
        assert SubscriptionTier.FREE.value == "free"


class TestBillingCycle:
    """Test the BillingCycle enum."""

    def test_cycles(self) -> None:
        assert BillingCycle.MONTHLY == "monthly"
        assert BillingCycle.ANNUAL == "annual"


class TestUnlimitedSentinel:
    """Test the UNLIMITED constant."""

    def test_unlimited_is_negative_one(self) -> None:
        assert UNLIMITED == -1

    def test_unlimited_is_int(self) -> None:
        assert isinstance(UNLIMITED, int)


class TestPricingConfig:
    """Test PricingConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = PricingConfig()
        assert config.monthly_price_usd == 0.0
        assert config.annual_price_usd == 0.0
        assert config.per_seat is False
        assert config.min_seats == 1
        assert config.max_seats == 1

    def test_frozen(self) -> None:
        config = PricingConfig()
        with pytest.raises(AttributeError):
            config.monthly_price_usd = 99.0  # type: ignore[misc]


class TestRateLimitConfig:
    """Test RateLimitConfig frozen dataclass."""

    def test_defaults(self) -> None:
        config = RateLimitConfig()
        assert config.requests_per_hour == 30
        assert config.burst_size == 10

    def test_frozen(self) -> None:
        config = RateLimitConfig()
        with pytest.raises(AttributeError):
            config.requests_per_hour = 999  # type: ignore[misc]


class TestFreeConfig:
    """Validate Free tier configuration."""

    def test_tier_identity(self) -> None:
        assert FREE_CONFIG.tier == SubscriptionTier.FREE
        assert FREE_CONFIG.display_name == "Free"

    def test_pricing_is_zero(self) -> None:
        assert FREE_CONFIG.pricing.monthly_price_usd == 0.0
        assert FREE_CONFIG.pricing.annual_price_usd == 0.0
        assert FREE_CONFIG.pricing.per_seat is False

    def test_usage_limits(self) -> None:
        assert FREE_CONFIG.shifts_per_month == 10
        assert FREE_CONFIG.active_sessions == 5
        assert FREE_CONFIG.brain_queries_per_month == 15
        assert FREE_CONFIG.tokens_per_month == 1_000_000
        assert FREE_CONFIG.retention_days == 30

    def test_rate_limit(self) -> None:
        assert FREE_CONFIG.rate_limit.requests_per_hour == 30
        assert FREE_CONFIG.rate_limit.burst_size == 10

    def test_feature_gates(self) -> None:
        assert FREE_CONFIG.cockpit_enabled is False
        assert FREE_CONFIG.cockpit_view_only is True
        assert FREE_CONFIG.smart_dispatch is False
        assert FREE_CONFIG.digest_daily is False
        assert FREE_CONFIG.knowledge_gap_alerts is False
        assert FREE_CONFIG.sso_enabled is False

    def test_reverse_trial(self) -> None:
        assert FREE_CONFIG.reverse_trial_days == 7

    def test_consensus_lifetime_trial(self) -> None:
        assert FREE_CONFIG.consensus_lifetime_trials == 1

    def test_no_team_seats(self) -> None:
        assert FREE_CONFIG.team_seats_included == 0
        assert FREE_CONFIG.team_roles_full_rbac is False

    def test_no_integrations(self) -> None:
        assert FREE_CONFIG.api_keys_max == 0
        assert FREE_CONFIG.webhooks_max == 0
        assert FREE_CONFIG.file_upload_mb == 5


class TestProConfig:
    """Validate Pro tier configuration."""

    def test_tier_identity(self) -> None:
        assert PRO_CONFIG.tier == SubscriptionTier.PRO
        assert PRO_CONFIG.display_name == "Pro"

    def test_pricing(self) -> None:
        assert PRO_CONFIG.pricing.monthly_price_usd == 16.0
        assert PRO_CONFIG.pricing.annual_price_usd == 144.0  # $12/mo
        assert PRO_CONFIG.pricing.per_seat is False

    def test_unlimited_core_resources(self) -> None:
        assert PRO_CONFIG.shifts_per_month == UNLIMITED
        assert PRO_CONFIG.active_sessions == UNLIMITED
        assert PRO_CONFIG.brain_queries_per_month == UNLIMITED
        assert PRO_CONFIG.auto_captures_per_month == UNLIMITED

    def test_capped_extras(self) -> None:
        assert PRO_CONFIG.tokens_per_month == 5_000_000
        assert PRO_CONFIG.retention_days == 180
        assert PRO_CONFIG.chain_steps_max == 5

    def test_rate_limit(self) -> None:
        assert PRO_CONFIG.rate_limit.requests_per_hour == 600
        assert PRO_CONFIG.rate_limit.burst_size == 50

    def test_four_team_seats(self) -> None:
        assert PRO_CONFIG.team_seats_included == 4

    def test_cockpit_enabled(self) -> None:
        assert PRO_CONFIG.cockpit_enabled is True
        assert PRO_CONFIG.cockpit_models_max == 2
        assert PRO_CONFIG.cockpit_view_only is False

    def test_no_consensus(self) -> None:
        assert PRO_CONFIG.consensus_models_max == 0

    def test_smart_dispatch(self) -> None:
        assert PRO_CONFIG.smart_dispatch is True

    def test_integrations(self) -> None:
        assert PRO_CONFIG.api_keys_max == 2
        assert PRO_CONFIG.webhooks_max == 3
        assert PRO_CONFIG.file_upload_mb == 25

    def test_no_sso(self) -> None:
        assert PRO_CONFIG.sso_enabled is False

    def test_support_level(self) -> None:
        assert PRO_CONFIG.support_level == "email"
        assert PRO_CONFIG.support_response_hours == 48


class TestProTeamConfig:
    """Validate Pro Team tier configuration."""

    def test_tier_identity(self) -> None:
        assert PRO_TEAM_CONFIG.tier == SubscriptionTier.PRO_TEAM
        assert PRO_TEAM_CONFIG.display_name == "Pro Team"

    def test_per_seat_pricing(self) -> None:
        assert PRO_TEAM_CONFIG.pricing.monthly_price_usd == 14.0
        assert PRO_TEAM_CONFIG.pricing.annual_price_usd == 108.0  # $9/seat/mo
        assert PRO_TEAM_CONFIG.pricing.per_seat is True
        assert PRO_TEAM_CONFIG.pricing.min_seats == 5
        assert PRO_TEAM_CONFIG.pricing.max_seats == 50

    def test_consensus_enabled(self) -> None:
        assert PRO_TEAM_CONFIG.consensus_models_max == 4

    def test_full_rbac(self) -> None:
        assert PRO_TEAM_CONFIG.team_roles_full_rbac is True

    def test_sso_enabled(self) -> None:
        assert PRO_TEAM_CONFIG.sso_enabled is True

    def test_unlimited_retention(self) -> None:
        assert PRO_TEAM_CONFIG.retention_days == UNLIMITED

    def test_cockpit_models(self) -> None:
        assert PRO_TEAM_CONFIG.cockpit_models_max == 4

    def test_rate_limit(self) -> None:
        assert PRO_TEAM_CONFIG.rate_limit.requests_per_hour == 2000
        assert PRO_TEAM_CONFIG.rate_limit.burst_size == 200


class TestEnterpriseConfig:
    """Validate Enterprise tier configuration."""

    def test_tier_identity(self) -> None:
        assert ENTERPRISE_CONFIG.tier == SubscriptionTier.ENTERPRISE
        assert ENTERPRISE_CONFIG.display_name == "Enterprise"

    def test_custom_pricing(self) -> None:
        assert ENTERPRISE_CONFIG.pricing.monthly_price_usd == 0.0  # Custom
        assert ENTERPRISE_CONFIG.pricing.per_seat is True
        assert ENTERPRISE_CONFIG.pricing.min_seats == 50
        assert ENTERPRISE_CONFIG.pricing.max_seats == UNLIMITED

    def test_everything_unlimited(self) -> None:
        assert ENTERPRISE_CONFIG.shifts_per_month == UNLIMITED
        assert ENTERPRISE_CONFIG.active_sessions == UNLIMITED
        assert ENTERPRISE_CONFIG.brain_queries_per_month == UNLIMITED
        assert ENTERPRISE_CONFIG.tokens_per_month == UNLIMITED
        assert ENTERPRISE_CONFIG.retention_days == UNLIMITED
        assert ENTERPRISE_CONFIG.auto_captures_per_month == UNLIMITED
        assert ENTERPRISE_CONFIG.chain_steps_max == UNLIMITED
        assert ENTERPRISE_CONFIG.consensus_models_max == UNLIMITED

    def test_unlimited_rate_limit(self) -> None:
        assert ENTERPRISE_CONFIG.rate_limit.requests_per_hour == UNLIMITED
        assert ENTERPRISE_CONFIG.rate_limit.burst_size == UNLIMITED

    def test_all_features_enabled(self) -> None:
        assert ENTERPRISE_CONFIG.cockpit_enabled is True
        assert ENTERPRISE_CONFIG.smart_dispatch is True
        assert ENTERPRISE_CONFIG.digest_daily is True
        assert ENTERPRISE_CONFIG.knowledge_gap_alerts is True
        assert ENTERPRISE_CONFIG.analytics_enabled is True
        assert ENTERPRISE_CONFIG.analytics_full is True
        assert ENTERPRISE_CONFIG.sso_enabled is True

    def test_dedicated_support(self) -> None:
        assert ENTERPRISE_CONFIG.support_level == "dedicated"
        assert ENTERPRISE_CONFIG.support_response_hours == 4


class TestGetTierConfig:
    """Test get_tier_config() lookup."""

    def test_free(self) -> None:
        config = get_tier_config(SubscriptionTier.FREE)
        assert config is FREE_CONFIG

    def test_pro(self) -> None:
        config = get_tier_config(SubscriptionTier.PRO)
        assert config is PRO_CONFIG

    def test_pro_team(self) -> None:
        config = get_tier_config(SubscriptionTier.PRO_TEAM)
        assert config is PRO_TEAM_CONFIG

    def test_enterprise(self) -> None:
        config = get_tier_config(SubscriptionTier.ENTERPRISE)
        assert config is ENTERPRISE_CONFIG

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tier"):
            get_tier_config("invalid")  # type: ignore[arg-type]


class TestGetAllTiers:
    """Test get_all_tiers() ordering."""

    def test_returns_four_tiers(self) -> None:
        tiers = get_all_tiers()
        assert len(tiers) == 4

    def test_display_order(self) -> None:
        tiers = get_all_tiers()
        assert tiers[0].tier == SubscriptionTier.FREE
        assert tiers[1].tier == SubscriptionTier.PRO
        assert tiers[2].tier == SubscriptionTier.PRO_TEAM
        assert tiers[3].tier == SubscriptionTier.ENTERPRISE

    def test_returns_list_of_tier_configs(self) -> None:
        tiers = get_all_tiers()
        for t in tiers:
            assert isinstance(t, TierConfig)


class TestIsWithinLimit:
    """Test is_within_limit() helper."""

    def test_under_limit(self) -> None:
        assert is_within_limit(5, 10) is True

    def test_at_limit(self) -> None:
        assert is_within_limit(10, 10) is False

    def test_over_limit(self) -> None:
        assert is_within_limit(15, 10) is False

    def test_zero_usage(self) -> None:
        assert is_within_limit(0, 10) is True

    def test_unlimited(self) -> None:
        assert is_within_limit(999_999, UNLIMITED) is True

    def test_zero_limit(self) -> None:
        assert is_within_limit(0, 0) is False


class TestRemainingQuota:
    """Test remaining_quota() helper."""

    def test_normal_remaining(self) -> None:
        assert remaining_quota(3, 10) == 7

    def test_at_limit(self) -> None:
        assert remaining_quota(10, 10) == 0

    def test_over_limit_returns_zero(self) -> None:
        assert remaining_quota(15, 10) == 0

    def test_unlimited(self) -> None:
        assert remaining_quota(999, UNLIMITED) == UNLIMITED

    def test_zero_usage(self) -> None:
        assert remaining_quota(0, 100) == 100


class TestTierConfigImmutability:
    """Verify TierConfig is frozen."""

    def test_cannot_mutate_shifts(self) -> None:
        with pytest.raises(AttributeError):
            FREE_CONFIG.shifts_per_month = 999  # type: ignore[misc]

    def test_cannot_mutate_tier(self) -> None:
        with pytest.raises(AttributeError):
            PRO_CONFIG.tier = SubscriptionTier.FREE  # type: ignore[misc]


class TestTierProgression:
    """Verify limits increase across tiers."""

    def test_rate_limits_increase(self) -> None:
        free_rate = FREE_CONFIG.rate_limit.requests_per_hour
        pro_rate = PRO_CONFIG.rate_limit.requests_per_hour
        team_rate = PRO_TEAM_CONFIG.rate_limit.requests_per_hour
        assert free_rate < pro_rate < team_rate

    def test_file_upload_increases(self) -> None:
        assert FREE_CONFIG.file_upload_mb < PRO_CONFIG.file_upload_mb
        assert PRO_CONFIG.file_upload_mb < PRO_TEAM_CONFIG.file_upload_mb
        assert PRO_TEAM_CONFIG.file_upload_mb < ENTERPRISE_CONFIG.file_upload_mb

    def test_api_keys_increase(self) -> None:
        assert FREE_CONFIG.api_keys_max < PRO_CONFIG.api_keys_max
        assert PRO_CONFIG.api_keys_max < PRO_TEAM_CONFIG.api_keys_max
        assert PRO_TEAM_CONFIG.api_keys_max < ENTERPRISE_CONFIG.api_keys_max
