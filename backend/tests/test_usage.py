"""Tests for UsageTracker — monthly counters and lifetime metrics.

Tests cover:
    - UsageMetric enum values
    - UsageSnapshot frozen dataclass
    - UsageCheckResult structure
    - increment() for monthly and lifetime metrics
    - get_usage() for known and unknown users
    - get_snapshot() completeness
    - check_within_limit() allowed and denied
    - check_within_limit() unlimited
    - reset_user() and reset_monthly()
    - _current_period() format
    - Multiple metrics per user
    - Multiple users isolation
"""

import pytest
from unittest.mock import patch

from app.core.billing.tiers import UNLIMITED
from app.core.billing.usage import (
    UsageCheckResult,
    UsageMetric,
    UsageSnapshot,
    UsageTracker,
    _current_period,
    _LIFETIME_METRICS,
)


class TestUsageMetric:
    """Test the UsageMetric enum."""

    def test_all_metrics_exist(self) -> None:
        assert UsageMetric.SHIFTS == "shifts"
        assert UsageMetric.BRAIN_QUERIES == "brain_queries"
        assert UsageMetric.TOKENS_PROCESSED == "tokens_processed"
        assert UsageMetric.AUTO_CAPTURES == "auto_captures"
        assert UsageMetric.CONSENSUS_TRIALS == "consensus_trials"
        assert UsageMetric.API_REQUESTS == "api_requests"

    def test_metric_count(self) -> None:
        assert len(UsageMetric) == 6


class TestLifetimeMetrics:
    """Test _LIFETIME_METRICS set."""

    def test_consensus_is_lifetime(self) -> None:
        assert UsageMetric.CONSENSUS_TRIALS in _LIFETIME_METRICS

    def test_shifts_are_not_lifetime(self) -> None:
        assert UsageMetric.SHIFTS not in _LIFETIME_METRICS


class TestCurrentPeriod:
    """Test the _current_period() helper."""

    def test_format_is_yyyy_mm(self) -> None:
        period = _current_period()
        parts = period.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 4  # YYYY
        assert len(parts[1]) == 2  # MM

    def test_month_is_zero_padded(self) -> None:
        from datetime import datetime, timezone

        with patch("app.core.billing.usage.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 1, 15, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            period = _current_period()
            assert period == "2026-01"


class TestUsageTrackerIncrement:
    """Test UsageTracker.increment()."""

    def test_increment_monthly_metric(self) -> None:
        tracker = UsageTracker()
        result = tracker.increment("user1", UsageMetric.SHIFTS, 1)
        assert result == 1

    def test_increment_accumulates(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 3)
        result = tracker.increment("user1", UsageMetric.SHIFTS, 2)
        assert result == 5

    def test_increment_default_amount(self) -> None:
        tracker = UsageTracker()
        result = tracker.increment("user1", UsageMetric.BRAIN_QUERIES)
        assert result == 1

    def test_increment_lifetime_metric(self) -> None:
        tracker = UsageTracker()
        result = tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        assert result == 1

    def test_lifetime_metric_accumulates(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        result = tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        assert result == 2

    def test_separate_users_are_isolated(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.increment("user2", UsageMetric.SHIFTS, 3)
        assert tracker.get_usage("user1", UsageMetric.SHIFTS) == 5
        assert tracker.get_usage("user2", UsageMetric.SHIFTS) == 3

    def test_separate_metrics_are_isolated(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.increment("user1", UsageMetric.BRAIN_QUERIES, 10)
        assert tracker.get_usage("user1", UsageMetric.SHIFTS) == 5
        assert tracker.get_usage("user1", UsageMetric.BRAIN_QUERIES) == 10


class TestUsageTrackerGetUsage:
    """Test UsageTracker.get_usage()."""

    def test_unknown_user_returns_zero(self) -> None:
        tracker = UsageTracker()
        assert tracker.get_usage("nobody", UsageMetric.SHIFTS) == 0

    def test_unknown_metric_returns_zero(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 1)
        assert tracker.get_usage("user1", UsageMetric.BRAIN_QUERIES) == 0

    def test_lifetime_unknown_user_returns_zero(self) -> None:
        tracker = UsageTracker()
        assert tracker.get_usage("nobody", UsageMetric.CONSENSUS_TRIALS) == 0

    def test_returns_current_count(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.TOKENS_PROCESSED, 50000)
        assert tracker.get_usage("user1", UsageMetric.TOKENS_PROCESSED) == 50000


class TestUsageTrackerGetSnapshot:
    """Test UsageTracker.get_snapshot()."""

    def test_empty_user_snapshot(self) -> None:
        tracker = UsageTracker()
        snap = tracker.get_snapshot("user1")
        assert snap.user_id == "user1"
        assert snap.metrics == {}
        assert snap.lifetime_metrics == {}
        assert isinstance(snap.period, str)

    def test_snapshot_includes_monthly(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.increment("user1", UsageMetric.BRAIN_QUERIES, 10)
        snap = tracker.get_snapshot("user1")
        assert snap.metrics[UsageMetric.SHIFTS] == 5
        assert snap.metrics[UsageMetric.BRAIN_QUERIES] == 10

    def test_snapshot_includes_lifetime(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        snap = tracker.get_snapshot("user1")
        assert snap.lifetime_metrics[UsageMetric.CONSENSUS_TRIALS] == 1

    def test_snapshot_is_frozen(self) -> None:
        tracker = UsageTracker()
        snap = tracker.get_snapshot("user1")
        with pytest.raises(AttributeError):
            snap.user_id = "hacked"  # type: ignore[misc]


class TestUsageTrackerCheckWithinLimit:
    """Test UsageTracker.check_within_limit()."""

    def test_under_limit_is_allowed(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 3)
        result = tracker.check_within_limit("user1", UsageMetric.SHIFTS, 10)
        assert result.allowed is True
        assert result.current == 3
        assert result.limit == 10
        assert result.remaining == 7

    def test_at_limit_is_denied(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 10)
        result = tracker.check_within_limit("user1", UsageMetric.SHIFTS, 10)
        assert result.allowed is False
        assert result.remaining == 0
        assert "limit reached" in result.message

    def test_over_limit_is_denied(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 15)
        result = tracker.check_within_limit("user1", UsageMetric.SHIFTS, 10)
        assert result.allowed is False
        assert result.remaining == 0

    def test_unlimited_is_always_allowed(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 999_999)
        result = tracker.check_within_limit("user1", UsageMetric.SHIFTS, UNLIMITED)
        assert result.allowed is True
        assert result.remaining == UNLIMITED
        assert "unlimited" in result.message

    def test_zero_usage_is_allowed(self) -> None:
        tracker = UsageTracker()
        result = tracker.check_within_limit("user1", UsageMetric.SHIFTS, 10)
        assert result.allowed is True
        assert result.remaining == 10

    def test_result_metric_matches(self) -> None:
        tracker = UsageTracker()
        result = tracker.check_within_limit("user1", UsageMetric.BRAIN_QUERIES, 15)
        assert result.metric == UsageMetric.BRAIN_QUERIES


class TestUsageTrackerReset:
    """Test reset methods."""

    def test_reset_user_clears_all(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        tracker.reset_user("user1")
        assert tracker.get_usage("user1", UsageMetric.SHIFTS) == 0
        assert tracker.get_usage("user1", UsageMetric.CONSENSUS_TRIALS) == 0

    def test_reset_user_doesnt_affect_others(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.increment("user2", UsageMetric.SHIFTS, 3)
        tracker.reset_user("user1")
        assert tracker.get_usage("user2", UsageMetric.SHIFTS) == 3

    def test_reset_monthly_clears_period(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.reset_monthly("user1")
        assert tracker.get_usage("user1", UsageMetric.SHIFTS) == 0

    def test_reset_monthly_preserves_lifetime(self) -> None:
        tracker = UsageTracker()
        tracker.increment("user1", UsageMetric.CONSENSUS_TRIALS, 1)
        tracker.increment("user1", UsageMetric.SHIFTS, 5)
        tracker.reset_monthly("user1")
        assert tracker.get_usage("user1", UsageMetric.CONSENSUS_TRIALS) == 1

    def test_reset_nonexistent_user_is_safe(self) -> None:
        tracker = UsageTracker()
        tracker.reset_user("nobody")  # Should not raise
        tracker.reset_monthly("nobody")  # Should not raise
