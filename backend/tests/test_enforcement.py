"""Tests for tier enforcement — FastAPI dependencies and error classes.

Tests cover:
    - PaywallError structure (402)
    - RateLimitError structure (429)
    - FeatureGateError structure (402)
    - resolve_user_tier() with and without trial
    - check_usage_limit() allowed and denied
    - check_rate_limit() allowed and denied
    - check_feature() enabled and disabled
    - record_usage() tracking
    - require_shifts() enforcement
    - require_brain_query() enforcement
    - require_cockpit() enforcement
    - require_consensus() with tier access, lifetime trial, and denial
    - require_smart_dispatch() enforcement
    - require_chain() enforcement and max steps return
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.core.billing.tiers import (
    UNLIMITED,
    SubscriptionTier,
    get_tier_config,
)
from app.core.billing.usage import UsageMetric, UsageTracker
from app.core.billing.trial import TrialManager
from app.core.billing.limiter import RateLimiter
from app.core.billing import enforcement


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Replace enforcement singletons with fresh instances for each test."""
    original_tracker = enforcement._usage_tracker
    original_limiter = enforcement._rate_limiter
    original_trial = enforcement._trial_manager

    enforcement._usage_tracker = UsageTracker()
    enforcement._rate_limiter = RateLimiter()
    enforcement._trial_manager = TrialManager()

    yield

    enforcement._usage_tracker = original_tracker
    enforcement._rate_limiter = original_limiter
    enforcement._trial_manager = original_trial


class TestPaywallError:
    """Test PaywallError (402)."""

    def test_status_code(self) -> None:
        err = enforcement.PaywallError(
            resource="shifts",
            current=10,
            limit=10,
            current_tier="free",
        )
        assert err.status_code == 402

    def test_detail_structure(self) -> None:
        err = enforcement.PaywallError(
            resource="shifts",
            current=10,
            limit=10,
            current_tier="free",
            message="You hit your limit!",
        )
        assert err.detail["error"] == "tier_limit_exceeded"
        assert err.detail["resource"] == "shifts"
        assert err.detail["current_usage"] == 10
        assert err.detail["limit"] == 10
        assert err.detail["current_tier"] == "free"
        assert err.detail["message"] == "You hit your limit!"
        assert err.detail["upgrade_url"] == "/pricing"

    def test_default_message(self) -> None:
        err = enforcement.PaywallError(
            resource="brain_queries",
            current=15,
            limit=15,
            current_tier="free",
        )
        assert "brain_queries" in err.detail["message"]
        assert "Upgrade" in err.detail["message"]


class TestRateLimitError:
    """Test RateLimitError (429)."""

    def test_status_code(self) -> None:
        from app.core.billing.limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=40,
            reset_in_seconds=5.0,
            retry_after=5.0,
        )
        err = enforcement.RateLimitError(result)
        assert err.status_code == 429

    def test_detail_includes_rate_info(self) -> None:
        from app.core.billing.limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=40,
            reset_in_seconds=5.0,
            retry_after=5.0,
        )
        err = enforcement.RateLimitError(result)
        assert err.detail["error"] == "rate_limit_exceeded"
        assert err.detail["remaining"] == 0
        assert err.detail["retry_after"] == 5.0

    def test_headers_set(self) -> None:
        from app.core.billing.limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=40,
            reset_in_seconds=5.0,
            retry_after=5.0,
        )
        err = enforcement.RateLimitError(result)
        assert "Retry-After" in err.headers
        assert "X-RateLimit-Limit" in err.headers


class TestFeatureGateError:
    """Test FeatureGateError (402)."""

    def test_status_code(self) -> None:
        err = enforcement.FeatureGateError(
            feature="cockpit_enabled",
            current_tier="free",
        )
        assert err.status_code == 402

    def test_detail(self) -> None:
        err = enforcement.FeatureGateError(
            feature="Smart Dispatch",
            current_tier="free",
            required_tier="Pro",
        )
        assert err.detail["error"] == "feature_not_available"
        assert err.detail["feature"] == "Smart Dispatch"
        assert err.detail["required_tier"] == "Pro"
        assert "Pro" in err.detail["message"]


class TestResolveUserTier:
    """Test resolve_user_tier()."""

    def test_free_user_no_trial(self) -> None:
        tier, config = enforcement.resolve_user_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.FREE
        assert config.tier == SubscriptionTier.FREE

    def test_free_user_with_active_trial(self) -> None:
        enforcement._trial_manager.start_trial("user1")
        tier, config = enforcement.resolve_user_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.PRO
        assert config.tier == SubscriptionTier.PRO

    def test_free_user_expired_trial(self) -> None:
        enforcement._trial_manager.start_trial("user1")
        enforcement._trial_manager.expire_trial("user1")
        tier, config = enforcement.resolve_user_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.FREE

    def test_pro_user(self) -> None:
        tier, config = enforcement.resolve_user_tier("user1", SubscriptionTier.PRO)
        assert tier == SubscriptionTier.PRO

    def test_enterprise_user(self) -> None:
        tier, config = enforcement.resolve_user_tier("user1", SubscriptionTier.ENTERPRISE)
        assert tier == SubscriptionTier.ENTERPRISE


class TestCheckUsageLimit:
    """Test check_usage_limit()."""

    def test_under_limit_passes(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 3)
        # Should not raise
        enforcement.check_usage_limit("user1", UsageMetric.SHIFTS, 10, "Free")

    def test_at_limit_raises_paywall(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 10)
        with pytest.raises(HTTPException) as exc_info:
            enforcement.check_usage_limit("user1", UsageMetric.SHIFTS, 10, "Free")
        assert exc_info.value.status_code == 402

    def test_unlimited_never_raises(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 999_999)
        # Should not raise
        enforcement.check_usage_limit("user1", UsageMetric.SHIFTS, UNLIMITED, "Pro")


class TestCheckRateLimit:
    """Test check_rate_limit()."""

    def test_within_rate_passes(self) -> None:
        config = get_tier_config(SubscriptionTier.PRO)
        # Should not raise — Pro has 600/hr + 50 burst
        enforcement.check_rate_limit("user1", config)

    def test_exhausted_rate_raises(self) -> None:
        config = get_tier_config(SubscriptionTier.FREE)
        # Free: 30/hr + 10 burst = 40 capacity
        for _ in range(40):
            enforcement.check_rate_limit("user1", config)
        with pytest.raises(HTTPException) as exc_info:
            enforcement.check_rate_limit("user1", config)
        assert exc_info.value.status_code == 429

    def test_unlimited_rate_never_raises(self) -> None:
        config = get_tier_config(SubscriptionTier.ENTERPRISE)
        for _ in range(100):
            enforcement.check_rate_limit("user1", config)


class TestCheckFeature:
    """Test check_feature()."""

    def test_enabled_feature_passes(self) -> None:
        config = get_tier_config(SubscriptionTier.PRO)
        # Should not raise
        enforcement.check_feature("cockpit_enabled", config, "Pro")

    def test_disabled_feature_raises(self) -> None:
        config = get_tier_config(SubscriptionTier.FREE)
        with pytest.raises(HTTPException) as exc_info:
            enforcement.check_feature("cockpit_enabled", config, "Free")
        assert exc_info.value.status_code == 402
        assert exc_info.value.detail["feature"] == "cockpit_enabled"

    def test_smart_dispatch_free_raises(self) -> None:
        config = get_tier_config(SubscriptionTier.FREE)
        with pytest.raises(HTTPException):
            enforcement.check_feature("smart_dispatch", config, "Free")


class TestRecordUsage:
    """Test record_usage()."""

    def test_records_and_returns_new_total(self) -> None:
        result = enforcement.record_usage("user1", UsageMetric.SHIFTS, 1)
        assert result == 1
        result = enforcement.record_usage("user1", UsageMetric.SHIFTS, 2)
        assert result == 3

    def test_records_different_metrics(self) -> None:
        enforcement.record_usage("user1", UsageMetric.SHIFTS, 5)
        enforcement.record_usage("user1", UsageMetric.BRAIN_QUERIES, 10)
        assert enforcement._usage_tracker.get_usage("user1", UsageMetric.SHIFTS) == 5
        assert enforcement._usage_tracker.get_usage("user1", UsageMetric.BRAIN_QUERIES) == 10


class TestRequireShifts:
    """Test require_shifts() dependency."""

    def test_free_user_under_limit(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 3)
        # Should not raise (free = 10 shifts)
        enforcement.require_shifts("user1", SubscriptionTier.FREE)

    def test_free_user_at_limit(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 10)
        with pytest.raises(HTTPException) as exc_info:
            enforcement.require_shifts("user1", SubscriptionTier.FREE)
        assert exc_info.value.status_code == 402

    def test_pro_user_unlimited(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.SHIFTS, 9999)
        # Should not raise (pro = unlimited)
        enforcement.require_shifts("user1", SubscriptionTier.PRO)


class TestRequireBrainQuery:
    """Test require_brain_query() dependency."""

    def test_free_user_under_limit(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.BRAIN_QUERIES, 5)
        enforcement.require_brain_query("user1", SubscriptionTier.FREE)

    def test_free_user_at_limit(self) -> None:
        enforcement._usage_tracker.increment("user1", UsageMetric.BRAIN_QUERIES, 15)
        with pytest.raises(HTTPException):
            enforcement.require_brain_query("user1", SubscriptionTier.FREE)


class TestRequireCockpit:
    """Test require_cockpit() dependency."""

    def test_pro_user_allowed(self) -> None:
        enforcement.require_cockpit("user1", SubscriptionTier.PRO)

    def test_free_user_denied(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            enforcement.require_cockpit("user1", SubscriptionTier.FREE)
        assert exc_info.value.status_code == 402


class TestRequireConsensus:
    """Test require_consensus() dependency."""

    def test_pro_team_allowed(self) -> None:
        # Pro Team has consensus_models_max=4
        enforcement.require_consensus("user1", SubscriptionTier.PRO_TEAM)

    def test_enterprise_allowed(self) -> None:
        enforcement.require_consensus("user1", SubscriptionTier.ENTERPRISE)

    def test_free_user_first_trial_allowed(self) -> None:
        # Free has consensus_lifetime_trials=1, 0 used
        enforcement.require_consensus("user1", SubscriptionTier.FREE)

    def test_free_user_trial_exhausted_denied(self) -> None:
        # Use up the 1 lifetime trial
        enforcement._usage_tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        with pytest.raises(HTTPException) as exc_info:
            enforcement.require_consensus("user1", SubscriptionTier.FREE)
        assert exc_info.value.status_code == 402

    def test_pro_user_no_consensus(self) -> None:
        # Pro has consensus_models_max=0, consensus_lifetime_trials=0
        with pytest.raises(HTTPException):
            enforcement.require_consensus("user1", SubscriptionTier.PRO)


class TestRequireSmartDispatch:
    """Test require_smart_dispatch() dependency."""

    def test_pro_allowed(self) -> None:
        enforcement.require_smart_dispatch("user1", SubscriptionTier.PRO)

    def test_free_denied(self) -> None:
        with pytest.raises(HTTPException):
            enforcement.require_smart_dispatch("user1", SubscriptionTier.FREE)


class TestRequireChain:
    """Test require_chain() dependency."""

    def test_pro_returns_max_steps(self) -> None:
        max_steps = enforcement.require_chain("user1", SubscriptionTier.PRO)
        assert max_steps == 5

    def test_pro_team_returns_more_steps(self) -> None:
        max_steps = enforcement.require_chain("user1", SubscriptionTier.PRO_TEAM)
        assert max_steps == 10

    def test_enterprise_returns_unlimited(self) -> None:
        max_steps = enforcement.require_chain("user1", SubscriptionTier.ENTERPRISE)
        assert max_steps == UNLIMITED

    def test_free_denied(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            enforcement.require_chain("user1", SubscriptionTier.FREE)
        assert exc_info.value.status_code == 402
        assert exc_info.value.detail["feature"] == "Chain Pipeline"


class TestGetterFunctions:
    """Test module-level getter functions."""

    def test_get_usage_tracker(self) -> None:
        tracker = enforcement.get_usage_tracker()
        assert isinstance(tracker, UsageTracker)

    def test_get_rate_limiter(self) -> None:
        limiter = enforcement.get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_trial_manager(self) -> None:
        mgr = enforcement.get_trial_manager()
        assert isinstance(mgr, TrialManager)
