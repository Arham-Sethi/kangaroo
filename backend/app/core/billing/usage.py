"""Usage tracking — monthly counters for metered resources.

Tracks per-user consumption of rate-limited resources:
    - Context shifts
    - Brain queries (search/recall)
    - Token processing
    - Auto-captures
    - Consensus trials (lifetime, not monthly)

Counters reset on the 1st of each month (UTC). The tracker is
backed by an in-memory store for development/testing and can be
swapped to Redis or PostgreSQL in production.

Usage:
    tracker = UsageTracker()
    tracker.increment(user_id, UsageMetric.SHIFTS, amount=1)
    current = tracker.get_usage(user_id, UsageMetric.SHIFTS)
    within = tracker.check_within_limit(user_id, UsageMetric.SHIFTS, limit=10)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class UsageMetric(str, Enum):
    """Metered usage resources."""

    SHIFTS = "shifts"
    BRAIN_QUERIES = "brain_queries"
    TOKENS_PROCESSED = "tokens_processed"
    AUTO_CAPTURES = "auto_captures"
    CONSENSUS_TRIALS = "consensus_trials"  # Lifetime, not monthly
    API_REQUESTS = "api_requests"  # Per-hour, handled separately


@dataclass(frozen=True)
class UsageSnapshot:
    """Point-in-time usage for a user.

    Attributes:
        user_id: The user whose usage this tracks.
        period: Month period in YYYY-MM format.
        metrics: Map of metric -> current count.
        lifetime_metrics: Metrics that don't reset monthly.
    """

    user_id: str
    period: str
    metrics: dict[UsageMetric, int]
    lifetime_metrics: dict[UsageMetric, int]


@dataclass(frozen=True)
class UsageCheckResult:
    """Result of checking whether usage is within limits.

    Attributes:
        allowed: Whether the action is allowed.
        metric: Which metric was checked.
        current: Current usage count.
        limit: The limit for this metric (-1 = unlimited).
        remaining: How many more are allowed (-1 = unlimited).
        message: Human-readable status message.
    """

    allowed: bool
    metric: UsageMetric
    current: int
    limit: int
    remaining: int
    message: str


def _current_period() -> str:
    """Get the current month period string (YYYY-MM)."""
    now = datetime.now(timezone.utc)
    return f"{now.year}-{now.month:02d}"


# Metrics that are lifetime (don't reset monthly)
_LIFETIME_METRICS = frozenset({UsageMetric.CONSENSUS_TRIALS})


class UsageTracker:
    """In-memory usage tracker with monthly reset.

    Production: Backed by Redis (atomic increments) or PostgreSQL.
    Development/Testing: In-memory dict.

    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self) -> None:
        # {user_id: {period: {metric: count}}}
        self._monthly: dict[str, dict[str, dict[UsageMetric, int]]] = {}
        # {user_id: {metric: count}} — lifetime metrics
        self._lifetime: dict[str, dict[UsageMetric, int]] = {}

    def increment(
        self,
        user_id: str,
        metric: UsageMetric,
        amount: int = 1,
    ) -> int:
        """Increment a usage counter.

        Args:
            user_id: The user to track.
            metric: Which metric to increment.
            amount: How much to add (default 1).

        Returns:
            The new count after incrementing.
        """
        if metric in _LIFETIME_METRICS:
            return self._increment_lifetime(user_id, metric, amount)
        return self._increment_monthly(user_id, metric, amount)

    def _increment_monthly(
        self, user_id: str, metric: UsageMetric, amount: int
    ) -> int:
        period = _current_period()

        if user_id not in self._monthly:
            self._monthly[user_id] = {}
        if period not in self._monthly[user_id]:
            self._monthly[user_id][period] = {}

        current = self._monthly[user_id][period].get(metric, 0)
        new_value = current + amount
        self._monthly[user_id][period][metric] = new_value
        return new_value

    def _increment_lifetime(
        self, user_id: str, metric: UsageMetric, amount: int
    ) -> int:
        if user_id not in self._lifetime:
            self._lifetime[user_id] = {}

        current = self._lifetime[user_id].get(metric, 0)
        new_value = current + amount
        self._lifetime[user_id][metric] = new_value
        return new_value

    def get_usage(self, user_id: str, metric: UsageMetric) -> int:
        """Get current usage for a metric.

        Returns 0 if no usage recorded.
        """
        if metric in _LIFETIME_METRICS:
            return self._lifetime.get(user_id, {}).get(metric, 0)

        period = _current_period()
        return (
            self._monthly.get(user_id, {}).get(period, {}).get(metric, 0)
        )

    def get_snapshot(self, user_id: str) -> UsageSnapshot:
        """Get a complete usage snapshot for a user."""
        period = _current_period()
        monthly = self._monthly.get(user_id, {}).get(period, {})
        lifetime = self._lifetime.get(user_id, {})

        return UsageSnapshot(
            user_id=user_id,
            period=period,
            metrics=dict(monthly),
            lifetime_metrics=dict(lifetime),
        )

    def check_within_limit(
        self,
        user_id: str,
        metric: UsageMetric,
        limit: int,
    ) -> UsageCheckResult:
        """Check if a user's usage is within the given limit.

        Args:
            user_id: The user to check.
            metric: Which metric to check.
            limit: The limit (-1 = unlimited).

        Returns:
            UsageCheckResult with allowed/denied status.
        """
        from app.core.billing.tiers import UNLIMITED

        current = self.get_usage(user_id, metric)

        if limit == UNLIMITED:
            return UsageCheckResult(
                allowed=True,
                metric=metric,
                current=current,
                limit=limit,
                remaining=UNLIMITED,
                message=f"{metric.value}: unlimited",
            )

        remaining = max(0, limit - current)
        allowed = current < limit

        if allowed:
            message = f"{metric.value}: {current}/{limit} used, {remaining} remaining"
        else:
            message = (
                f"{metric.value}: limit reached ({current}/{limit}). "
                f"Upgrade your plan for more."
            )

        return UsageCheckResult(
            allowed=allowed,
            metric=metric,
            current=current,
            limit=limit,
            remaining=remaining,
            message=message,
        )

    def reset_user(self, user_id: str) -> None:
        """Reset all usage for a user (monthly and lifetime)."""
        self._monthly.pop(user_id, None)
        self._lifetime.pop(user_id, None)

    def reset_monthly(self, user_id: str) -> None:
        """Reset only monthly usage for a user."""
        period = _current_period()
        if user_id in self._monthly:
            self._monthly[user_id].pop(period, None)
