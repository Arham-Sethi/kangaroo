"""Reverse trial system — 7-day full Pro access for new signups.

New users automatically get Pro-level access for 7 days.
After the trial expires, they downgrade to the Free tier.

The reverse trial exploits loss aversion: users experience full
Pro features, build data and habits, then feel the downgrade.
Research shows 7-21% conversion vs 2.6% for standard freemium.

Usage:
    manager = TrialManager()
    manager.start_trial(user_id)
    status = manager.get_trial_status(user_id)
    effective = manager.get_effective_tier(user_id, base_tier)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.billing.tiers import SubscriptionTier


@dataclass(frozen=True)
class TrialStatus:
    """Current state of a user's reverse trial.

    Attributes:
        user_id: The user.
        is_active: Whether the trial is currently active.
        started_at: Unix timestamp of trial start.
        expires_at: Unix timestamp of trial expiration.
        days_remaining: Days left in the trial (0 if expired).
        trial_tier: What tier the trial provides.
        has_been_offered: Whether this user was ever offered a trial.
    """

    user_id: str
    is_active: bool
    started_at: float
    expires_at: float
    days_remaining: float
    trial_tier: SubscriptionTier
    has_been_offered: bool


@dataclass(frozen=True)
class TrialConfig:
    """Configuration for the reverse trial.

    Attributes:
        trial_duration_days: How long the trial lasts.
        trial_tier: What tier the trial provides.
        allow_restart: Whether expired trials can be restarted.
    """

    trial_duration_days: int = 7
    trial_tier: SubscriptionTier = SubscriptionTier.PRO
    allow_restart: bool = False


class TrialManager:
    """Manages reverse trial lifecycle for users.

    In-memory for development. Production: Redis or DB-backed.
    """

    def __init__(self, config: TrialConfig | None = None) -> None:
        self._config = config or TrialConfig()
        # {user_id: (started_at, expires_at)}
        self._trials: dict[str, tuple[float, float]] = {}

    @property
    def config(self) -> TrialConfig:
        return self._config

    def start_trial(self, user_id: str) -> TrialStatus:
        """Start a reverse trial for a user.

        If the user already has/had a trial, returns existing status
        unless allow_restart is True.

        Args:
            user_id: The user starting the trial.

        Returns:
            TrialStatus with the trial state.
        """
        existing = self._trials.get(user_id)

        if existing is not None and not self._config.allow_restart:
            return self.get_trial_status(user_id)

        now = time.time()
        duration_seconds = self._config.trial_duration_days * 86400
        expires_at = now + duration_seconds

        self._trials[user_id] = (now, expires_at)

        return TrialStatus(
            user_id=user_id,
            is_active=True,
            started_at=now,
            expires_at=expires_at,
            days_remaining=self._config.trial_duration_days,
            trial_tier=self._config.trial_tier,
            has_been_offered=True,
        )

    def get_trial_status(self, user_id: str) -> TrialStatus:
        """Get the current trial status for a user.

        Returns inactive status if no trial has been started.
        """
        entry = self._trials.get(user_id)

        if entry is None:
            return TrialStatus(
                user_id=user_id,
                is_active=False,
                started_at=0.0,
                expires_at=0.0,
                days_remaining=0.0,
                trial_tier=self._config.trial_tier,
                has_been_offered=False,
            )

        started_at, expires_at = entry
        now = time.time()
        is_active = now < expires_at
        days_remaining = max(0.0, (expires_at - now) / 86400)

        return TrialStatus(
            user_id=user_id,
            is_active=is_active,
            started_at=started_at,
            expires_at=expires_at,
            days_remaining=round(days_remaining, 2),
            trial_tier=self._config.trial_tier,
            has_been_offered=True,
        )

    def get_effective_tier(
        self,
        user_id: str,
        base_tier: SubscriptionTier,
    ) -> SubscriptionTier:
        """Determine the effective tier considering trial status.

        If the user has an active trial and their base tier is FREE,
        they get elevated to the trial tier. Paid users are unaffected.

        Args:
            user_id: The user to check.
            base_tier: The user's actual subscription tier.

        Returns:
            The effective tier (may be elevated by trial).
        """
        # Paid users aren't affected by trials
        if base_tier != SubscriptionTier.FREE:
            return base_tier

        status = self.get_trial_status(user_id)
        if status.is_active:
            return status.trial_tier

        return base_tier

    def expire_trial(self, user_id: str) -> bool:
        """Manually expire a user's trial (for testing).

        Returns True if a trial was found and expired.
        """
        entry = self._trials.get(user_id)
        if entry is None:
            return False

        started_at, _ = entry
        self._trials[user_id] = (started_at, 0.0)  # Expired
        return True

    def has_trial_expired(self, user_id: str) -> bool:
        """Check if a user's trial has expired (had one, now over)."""
        status = self.get_trial_status(user_id)
        return status.has_been_offered and not status.is_active

    def active_trial_count(self) -> int:
        """Count currently active trials."""
        now = time.time()
        return sum(
            1 for _, expires_at in self._trials.values()
            if now < expires_at
        )
