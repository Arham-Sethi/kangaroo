"""Temporal decay — recency scoring and importance decay over time.

Older memories fade unless reinforced. This module computes time-weighted
importance scores so that:
    - Recent sessions carry more weight than old ones
    - Frequently referenced entities resist decay
    - "Current project" entities get a recency boost

Decay formula:
    decayed_importance = base_importance * decay_factor(age_days)
    decay_factor = exp(-lambda * age_days)

    lambda = ln(2) / half_life_days  (half-life decay)

Usage:
    decay = TemporalDecay(half_life_days=30)
    score = decay.compute(importance=0.9, age_days=15)  # ~0.636
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for temporal decay.

    Attributes:
        half_life_days: Days until importance halves (default 30).
        min_importance: Floor — entities never decay below this.
        reinforcement_boost: Extra importance per reinforcement event.
        max_importance: Ceiling for importance scores.
    """

    half_life_days: float = 30.0
    min_importance: float = 0.05
    reinforcement_boost: float = 0.1
    max_importance: float = 1.0


@dataclass(frozen=True)
class DecayedScore:
    """Result of applying temporal decay.

    Attributes:
        original: The base importance score.
        decayed: The time-weighted importance.
        age_days: How old the memory is.
        decay_factor: The multiplier applied (0..1).
    """

    original: float
    decayed: float
    age_days: float
    decay_factor: float


class TemporalDecay:
    """Applies time-based decay to entity and decision importance scores."""

    def __init__(self, config: DecayConfig | None = None) -> None:
        self._config = config or DecayConfig()
        # Decay constant: lambda = ln(2) / half_life
        self._lambda = math.log(2) / self._config.half_life_days

    @property
    def config(self) -> DecayConfig:
        return self._config

    def compute(
        self,
        importance: float,
        age_days: float,
        reinforcement_count: int = 0,
    ) -> DecayedScore:
        """Compute decayed importance for a single item.

        Args:
            importance: Base importance score (0..1).
            age_days: Days since the memory was created.
            reinforcement_count: How many times this was re-referenced.

        Returns:
            DecayedScore with the time-weighted importance.
        """
        # Reinforcement boosts base importance before decay
        boosted = importance + (
            reinforcement_count * self._config.reinforcement_boost
        )
        boosted = min(boosted, self._config.max_importance)

        # Exponential decay
        decay_factor = math.exp(-self._lambda * max(age_days, 0))

        decayed = boosted * decay_factor
        decayed = max(decayed, self._config.min_importance)
        decayed = min(decayed, self._config.max_importance)

        return DecayedScore(
            original=importance,
            decayed=round(decayed, 6),
            age_days=age_days,
            decay_factor=round(decay_factor, 6),
        )

    def compute_age_days(
        self,
        created_at: datetime,
        now: datetime | None = None,
    ) -> float:
        """Compute age in fractional days from a timestamp.

        Args:
            created_at: When the memory was created.
            now: Current time (defaults to utcnow).

        Returns:
            Age in fractional days (e.g., 1.5 = 36 hours).
        """
        if now is None:
            now = datetime.now(timezone.utc)
        # Ensure both are aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        delta = now - created_at
        return delta.total_seconds() / 86400.0

    def batch_compute(
        self,
        items: list[dict[str, Any]],
        importance_key: str = "importance",
        age_key: str = "age_days",
        reinforcement_key: str = "reinforcement_count",
    ) -> list[DecayedScore]:
        """Apply decay to a batch of items.

        Each item is a dict with at least `importance_key` and `age_key`.

        Returns:
            List of DecayedScore in the same order.
        """
        return [
            self.compute(
                importance=item[importance_key],
                age_days=item[age_key],
                reinforcement_count=item.get(reinforcement_key, 0),
            )
            for item in items
        ]
