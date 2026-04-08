"""Tests for billing API endpoints.

Tests cover:
    - GET /api/v1/billing/tiers — public tier listing
    - GET /api/v1/billing/current — subscription status
    - GET /api/v1/billing/usage — usage breakdown
    - POST /api/v1/billing/trial/start — start trial
    - GET /api/v1/billing/trial/status — trial status
    - GET /api/v1/billing/limits — tier limits
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.core.billing.tiers import (
    UNLIMITED,
    SubscriptionTier,
    get_all_tiers,
    get_tier_config,
)
from app.core.billing.usage import UsageMetric, UsageTracker
from app.core.billing.trial import TrialManager, TrialStatus
from app.core.billing.enforcement import (
    get_trial_manager,
    get_usage_tracker,
    resolve_user_tier,
)
from app.api.v1.billing import (
    AllTiersResponse,
    LimitsResponse,
    SubscriptionResponse,
    TierResponse,
    TrialStatusResponse,
    UsageMetricResponse,
    UsageResponse,
    list_tiers,
    get_current_subscription,
    get_usage,
    start_trial,
    get_trial_status,
    get_limits,
)


# ── Schema Tests ────────────────────────────────────────────────────────────


class TestTierResponse:
    """Test TierResponse schema structure."""

    def test_fields_populated(self) -> None:
        resp = TierResponse(
            tier="pro",
            display_name="Pro",
            pricing={"monthly_price_usd": 16.0, "annual_price_usd": 144.0,
                     "annual_monthly_equivalent": 12.0, "per_seat": False,
                     "min_seats": 1, "max_seats": 1},
            shifts_per_month=-1,
            active_sessions=-1,
            brain_queries_per_month=-1,
            tokens_per_month=5_000_000,
            retention_days=180,
            cockpit_models_max=2,
            chain_steps_max=5,
            consensus_models_max=0,
            team_seats_included=4,
            api_keys_max=2,
            webhooks_max=3,
            file_upload_mb=25,
            smart_dispatch=True,
            digest_daily=True,
            knowledge_gap_alerts=True,
            sso_enabled=False,
            support_level="email",
        )
        assert resp.tier == "pro"
        assert resp.display_name == "Pro"
        assert resp.shifts_per_month == -1


class TestUsageMetricResponse:
    """Test UsageMetricResponse schema."""

    def test_unlimited_metric(self) -> None:
        resp = UsageMetricResponse(
            metric="shifts",
            current=50,
            limit=-1,
            remaining=-1,
            is_unlimited=True,
        )
        assert resp.is_unlimited is True
        assert resp.remaining == -1

    def test_limited_metric(self) -> None:
        resp = UsageMetricResponse(
            metric="shifts",
            current=5,
            limit=10,
            remaining=5,
            is_unlimited=False,
        )
        assert resp.is_unlimited is False
        assert resp.remaining == 5


# ── Endpoint Unit Tests (direct function calls) ────────────────────────────


class TestListTiersEndpoint:
    """Test list_tiers() endpoint."""

    @pytest.mark.asyncio
    async def test_returns_four_tiers(self) -> None:
        result = await list_tiers()
        assert len(result.tiers) == 4

    @pytest.mark.asyncio
    async def test_tier_order(self) -> None:
        result = await list_tiers()
        tier_names = [t.tier for t in result.tiers]
        assert tier_names == ["free", "pro", "pro_team", "enterprise"]

    @pytest.mark.asyncio
    async def test_free_tier_pricing(self) -> None:
        result = await list_tiers()
        free = result.tiers[0]
        assert free.pricing.monthly_price_usd == 0.0
        assert free.pricing.annual_price_usd == 0.0

    @pytest.mark.asyncio
    async def test_pro_tier_pricing(self) -> None:
        result = await list_tiers()
        pro = result.tiers[1]
        assert pro.pricing.monthly_price_usd == 16.0
        assert pro.pricing.annual_price_usd == 144.0
        assert pro.pricing.annual_monthly_equivalent == 12.0

    @pytest.mark.asyncio
    async def test_pro_team_per_seat(self) -> None:
        result = await list_tiers()
        team = result.tiers[2]
        assert team.pricing.per_seat is True
        assert team.pricing.min_seats == 5
        assert team.pricing.max_seats == 50

    @pytest.mark.asyncio
    async def test_enterprise_zero_annual_monthly(self) -> None:
        result = await list_tiers()
        ent = result.tiers[3]
        # Enterprise has custom pricing (0.0 annual)
        assert ent.pricing.annual_monthly_equivalent == 0.0

    @pytest.mark.asyncio
    async def test_each_tier_has_all_fields(self) -> None:
        result = await list_tiers()
        for tier in result.tiers:
            assert isinstance(tier.tier, str)
            assert isinstance(tier.display_name, str)
            assert isinstance(tier.shifts_per_month, int)
            assert isinstance(tier.support_level, str)


class TestGetCurrentSubscription:
    """Test get_current_subscription() endpoint."""

    @pytest.mark.asyncio
    async def test_free_user_no_trial(self) -> None:
        user = MagicMock()
        user.id = "user-123"
        user.subscription_tier = "free"

        with patch("app.api.v1.billing.get_trial_manager") as mock_trial, \
             patch("app.api.v1.billing.resolve_user_tier") as mock_resolve:
            mock_trial.return_value.get_trial_status.return_value = TrialStatus(
                user_id="user-123",
                is_active=False,
                started_at=0.0,
                expires_at=0.0,
                days_remaining=0.0,
                trial_tier=SubscriptionTier.PRO,
                has_been_offered=False,
            )
            mock_resolve.return_value = (
                SubscriptionTier.FREE,
                get_tier_config(SubscriptionTier.FREE),
            )

            result = await get_current_subscription(user=user)
            assert result.tier == "free"
            assert result.effective_tier == "free"
            assert result.is_trial_active is False
            assert result.trial_expired is False

    @pytest.mark.asyncio
    async def test_free_user_active_trial(self) -> None:
        user = MagicMock()
        user.id = "user-456"
        user.subscription_tier = "free"

        with patch("app.api.v1.billing.get_trial_manager") as mock_trial, \
             patch("app.api.v1.billing.resolve_user_tier") as mock_resolve:
            mock_trial.return_value.get_trial_status.return_value = TrialStatus(
                user_id="user-456",
                is_active=True,
                started_at=1000.0,
                expires_at=9999999999.0,
                days_remaining=6.5,
                trial_tier=SubscriptionTier.PRO,
                has_been_offered=True,
            )
            mock_resolve.return_value = (
                SubscriptionTier.PRO,
                get_tier_config(SubscriptionTier.PRO),
            )

            result = await get_current_subscription(user=user)
            assert result.tier == "free"
            assert result.effective_tier == "pro"
            assert result.is_trial_active is True
            assert result.trial_days_remaining == 6.5

    @pytest.mark.asyncio
    async def test_expired_trial(self) -> None:
        user = MagicMock()
        user.id = "user-789"
        user.subscription_tier = "free"

        with patch("app.api.v1.billing.get_trial_manager") as mock_trial, \
             patch("app.api.v1.billing.resolve_user_tier") as mock_resolve:
            mock_trial.return_value.get_trial_status.return_value = TrialStatus(
                user_id="user-789",
                is_active=False,
                started_at=1000.0,
                expires_at=2000.0,
                days_remaining=0.0,
                trial_tier=SubscriptionTier.PRO,
                has_been_offered=True,
            )
            mock_resolve.return_value = (
                SubscriptionTier.FREE,
                get_tier_config(SubscriptionTier.FREE),
            )

            result = await get_current_subscription(user=user)
            assert result.trial_expired is True


class TestStartTrialEndpoint:
    """Test start_trial() endpoint."""

    @pytest.mark.asyncio
    async def test_start_returns_active(self) -> None:
        user = MagicMock()
        user.id = "user-new"

        with patch("app.api.v1.billing.get_trial_manager") as mock_trial:
            mock_trial.return_value.start_trial.return_value = TrialStatus(
                user_id="user-new",
                is_active=True,
                started_at=1000.0,
                expires_at=605800.0,
                days_remaining=7.0,
                trial_tier=SubscriptionTier.PRO,
                has_been_offered=True,
            )

            result = await start_trial(user=user)
            assert result.is_active is True
            assert result.days_remaining == 7.0
            assert result.trial_tier == "pro"


class TestGetTrialStatusEndpoint:
    """Test get_trial_status() endpoint."""

    @pytest.mark.asyncio
    async def test_no_trial(self) -> None:
        user = MagicMock()
        user.id = "user-notrial"

        with patch("app.api.v1.billing.get_trial_manager") as mock_trial:
            mock_trial.return_value.get_trial_status.return_value = TrialStatus(
                user_id="user-notrial",
                is_active=False,
                started_at=0.0,
                expires_at=0.0,
                days_remaining=0.0,
                trial_tier=SubscriptionTier.PRO,
                has_been_offered=False,
            )

            result = await get_trial_status(user=user)
            assert result.is_active is False
            assert result.has_been_offered is False


class TestGetUsageEndpoint:
    """Test get_usage() endpoint."""

    @pytest.mark.asyncio
    async def test_returns_metrics(self) -> None:
        user = MagicMock()
        user.id = "user-usage"
        user.subscription_tier = "free"

        mock_snapshot = MagicMock()
        mock_snapshot.period = "2026-04"
        mock_snapshot.metrics = {
            UsageMetric.SHIFTS: 5,
            UsageMetric.BRAIN_QUERIES: 10,
            UsageMetric.TOKENS_PROCESSED: 500000,
            UsageMetric.AUTO_CAPTURES: 0,
        }
        mock_snapshot.lifetime_metrics = {
            UsageMetric.CONSENSUS_TRIALS: 0,
        }

        with patch("app.api.v1.billing.resolve_user_tier") as mock_resolve, \
             patch("app.api.v1.billing.get_usage_tracker") as mock_tracker:
            config = get_tier_config(SubscriptionTier.FREE)
            mock_resolve.return_value = (SubscriptionTier.FREE, config)
            mock_tracker.return_value.get_snapshot.return_value = mock_snapshot

            result = await get_usage(user=user)
            assert result.tier == "free"
            assert result.period == "2026-04"
            assert len(result.metrics) == 5  # 4 monthly + 1 lifetime


class TestGetLimitsEndpoint:
    """Test get_limits() endpoint."""

    @pytest.mark.asyncio
    async def test_free_limits(self) -> None:
        user = MagicMock()
        user.id = "user-limits"
        user.subscription_tier = "free"

        with patch("app.api.v1.billing.resolve_user_tier") as mock_resolve:
            config = get_tier_config(SubscriptionTier.FREE)
            mock_resolve.return_value = (SubscriptionTier.FREE, config)

            result = await get_limits(user=user)
            assert result.tier == "free"
            assert result.display_name == "Free"
            assert result.limits["shifts_per_month"] == 10
            assert result.limits["cockpit_enabled"] is False
            assert result.limits["smart_dispatch"] is False

    @pytest.mark.asyncio
    async def test_pro_unlimited_shows_unlimited(self) -> None:
        user = MagicMock()
        user.id = "user-pro"
        user.subscription_tier = "pro"

        with patch("app.api.v1.billing.resolve_user_tier") as mock_resolve:
            config = get_tier_config(SubscriptionTier.PRO)
            mock_resolve.return_value = (SubscriptionTier.PRO, config)

            result = await get_limits(user=user)
            assert result.limits["shifts_per_month"] == "unlimited"
            assert result.limits["active_sessions"] == "unlimited"
            assert result.limits["cockpit_enabled"] is True
