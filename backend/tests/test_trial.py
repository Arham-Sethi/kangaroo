"""Tests for TrialManager — reverse trial lifecycle.

Tests cover:
    - TrialStatus frozen dataclass
    - TrialConfig defaults and custom values
    - start_trial() creates active trial
    - start_trial() idempotent (no restart by default)
    - start_trial() with allow_restart
    - get_trial_status() for new/active/expired users
    - get_effective_tier() elevation for free users
    - get_effective_tier() no effect on paid users
    - expire_trial() manual expiration
    - has_trial_expired() detection
    - active_trial_count() tracking
"""

import time
from unittest.mock import patch

import pytest

from app.core.billing.tiers import SubscriptionTier
from app.core.billing.trial import (
    TrialConfig,
    TrialManager,
    TrialStatus,
)


class TestTrialStatus:
    """Test TrialStatus frozen dataclass."""

    def test_create_status(self) -> None:
        status = TrialStatus(
            user_id="user1",
            is_active=True,
            started_at=1000.0,
            expires_at=2000.0,
            days_remaining=7.0,
            trial_tier=SubscriptionTier.PRO,
            has_been_offered=True,
        )
        assert status.user_id == "user1"
        assert status.is_active is True

    def test_frozen(self) -> None:
        status = TrialStatus(
            user_id="user1",
            is_active=True,
            started_at=0.0,
            expires_at=0.0,
            days_remaining=0.0,
            trial_tier=SubscriptionTier.PRO,
            has_been_offered=True,
        )
        with pytest.raises(AttributeError):
            status.is_active = False  # type: ignore[misc]


class TestTrialConfig:
    """Test TrialConfig defaults."""

    def test_defaults(self) -> None:
        config = TrialConfig()
        assert config.trial_duration_days == 7
        assert config.trial_tier == SubscriptionTier.PRO
        assert config.allow_restart is False

    def test_custom_config(self) -> None:
        config = TrialConfig(
            trial_duration_days=14,
            trial_tier=SubscriptionTier.PRO_TEAM,
            allow_restart=True,
        )
        assert config.trial_duration_days == 14
        assert config.trial_tier == SubscriptionTier.PRO_TEAM
        assert config.allow_restart is True


class TestTrialManagerStartTrial:
    """Test TrialManager.start_trial()."""

    def test_start_creates_active_trial(self) -> None:
        mgr = TrialManager()
        status = mgr.start_trial("user1")
        assert status.is_active is True
        assert status.has_been_offered is True
        assert status.days_remaining == 7
        assert status.trial_tier == SubscriptionTier.PRO

    def test_start_sets_timestamps(self) -> None:
        mgr = TrialManager()
        before = time.time()
        status = mgr.start_trial("user1")
        after = time.time()
        assert before <= status.started_at <= after
        assert status.expires_at > status.started_at
        expected_duration = 7 * 86400
        assert abs((status.expires_at - status.started_at) - expected_duration) < 1.0

    def test_start_is_idempotent(self) -> None:
        mgr = TrialManager()
        first = mgr.start_trial("user1")
        second = mgr.start_trial("user1")
        # Same started_at — didn't restart
        assert first.started_at == second.started_at

    def test_start_with_allow_restart(self) -> None:
        config = TrialConfig(allow_restart=True)
        mgr = TrialManager(config)
        first = mgr.start_trial("user1")
        # Expire the trial
        mgr.expire_trial("user1")
        second = mgr.start_trial("user1")
        # New trial started
        assert second.started_at > first.started_at
        assert second.is_active is True

    def test_separate_users(self) -> None:
        mgr = TrialManager()
        s1 = mgr.start_trial("user1")
        s2 = mgr.start_trial("user2")
        assert s1.user_id == "user1"
        assert s2.user_id == "user2"
        assert s1.is_active is True
        assert s2.is_active is True


class TestTrialManagerGetStatus:
    """Test TrialManager.get_trial_status()."""

    def test_no_trial_returns_inactive(self) -> None:
        mgr = TrialManager()
        status = mgr.get_trial_status("nobody")
        assert status.is_active is False
        assert status.has_been_offered is False
        assert status.started_at == 0.0
        assert status.expires_at == 0.0
        assert status.days_remaining == 0.0

    def test_active_trial_status(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        status = mgr.get_trial_status("user1")
        assert status.is_active is True
        assert status.has_been_offered is True
        assert status.days_remaining > 0

    def test_expired_trial_status(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        mgr.expire_trial("user1")
        status = mgr.get_trial_status("user1")
        assert status.is_active is False
        assert status.has_been_offered is True
        assert status.days_remaining == 0.0


class TestTrialManagerEffectiveTier:
    """Test TrialManager.get_effective_tier()."""

    def test_free_user_elevated_during_trial(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        tier = mgr.get_effective_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.PRO

    def test_free_user_not_elevated_after_expiry(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        mgr.expire_trial("user1")
        tier = mgr.get_effective_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.FREE

    def test_pro_user_unaffected_by_trial(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        tier = mgr.get_effective_tier("user1", SubscriptionTier.PRO)
        assert tier == SubscriptionTier.PRO

    def test_pro_team_user_unaffected(self) -> None:
        mgr = TrialManager()
        tier = mgr.get_effective_tier("user1", SubscriptionTier.PRO_TEAM)
        assert tier == SubscriptionTier.PRO_TEAM

    def test_enterprise_user_unaffected(self) -> None:
        mgr = TrialManager()
        tier = mgr.get_effective_tier("user1", SubscriptionTier.ENTERPRISE)
        assert tier == SubscriptionTier.ENTERPRISE

    def test_free_user_without_trial(self) -> None:
        mgr = TrialManager()
        tier = mgr.get_effective_tier("user1", SubscriptionTier.FREE)
        assert tier == SubscriptionTier.FREE


class TestTrialManagerExpire:
    """Test TrialManager.expire_trial()."""

    def test_expire_active_trial(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        result = mgr.expire_trial("user1")
        assert result is True
        status = mgr.get_trial_status("user1")
        assert status.is_active is False

    def test_expire_nonexistent_trial(self) -> None:
        mgr = TrialManager()
        result = mgr.expire_trial("nobody")
        assert result is False


class TestTrialManagerHasExpired:
    """Test TrialManager.has_trial_expired()."""

    def test_no_trial_not_expired(self) -> None:
        mgr = TrialManager()
        assert mgr.has_trial_expired("nobody") is False

    def test_active_trial_not_expired(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        assert mgr.has_trial_expired("user1") is False

    def test_expired_trial_is_expired(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        mgr.expire_trial("user1")
        assert mgr.has_trial_expired("user1") is True


class TestTrialManagerActiveCount:
    """Test TrialManager.active_trial_count()."""

    def test_no_trials(self) -> None:
        mgr = TrialManager()
        assert mgr.active_trial_count() == 0

    def test_one_active(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        assert mgr.active_trial_count() == 1

    def test_mixed_active_and_expired(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        mgr.start_trial("user2")
        mgr.expire_trial("user1")
        assert mgr.active_trial_count() == 1

    def test_all_expired(self) -> None:
        mgr = TrialManager()
        mgr.start_trial("user1")
        mgr.start_trial("user2")
        mgr.expire_trial("user1")
        mgr.expire_trial("user2")
        assert mgr.active_trial_count() == 0


class TestTrialManagerConfig:
    """Test TrialManager.config property."""

    def test_default_config(self) -> None:
        mgr = TrialManager()
        assert mgr.config.trial_duration_days == 7
        assert mgr.config.trial_tier == SubscriptionTier.PRO

    def test_custom_config(self) -> None:
        config = TrialConfig(trial_duration_days=14)
        mgr = TrialManager(config)
        assert mgr.config.trial_duration_days == 14
