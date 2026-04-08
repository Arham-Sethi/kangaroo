"""Tests for TemporalDecay — recency scoring and importance decay.

Tests cover:
    - Basic exponential decay formula
    - Half-life behavior (importance halves at half_life_days)
    - Minimum importance floor
    - Reinforcement boost
    - Age computation from timestamps
    - Batch computation
    - Zero age (no decay)
    - Very old memories
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.core.brain.temporal import DecayConfig, DecayedScore, TemporalDecay


class TestTemporalDecayBasic:
    def test_no_decay_at_zero_age(self) -> None:
        decay = TemporalDecay()
        result = decay.compute(importance=0.8, age_days=0)
        assert result.decayed == 0.8
        assert result.decay_factor == 1.0

    def test_half_life_halves_importance(self) -> None:
        config = DecayConfig(half_life_days=30)
        decay = TemporalDecay(config)
        result = decay.compute(importance=1.0, age_days=30)
        assert abs(result.decayed - 0.5) < 0.01

    def test_two_half_lives_quarters_importance(self) -> None:
        config = DecayConfig(half_life_days=30)
        decay = TemporalDecay(config)
        result = decay.compute(importance=1.0, age_days=60)
        assert abs(result.decayed - 0.25) < 0.01

    def test_decay_factor_range(self) -> None:
        decay = TemporalDecay()
        result = decay.compute(importance=0.9, age_days=15)
        assert 0 < result.decay_factor <= 1.0

    def test_result_is_frozen(self) -> None:
        decay = TemporalDecay()
        result = decay.compute(importance=0.5, age_days=10)
        assert isinstance(result, DecayedScore)
        with pytest.raises(AttributeError):
            result.decayed = 0.99  # type: ignore[misc]

    def test_very_old_memory_reaches_floor(self) -> None:
        config = DecayConfig(half_life_days=30, min_importance=0.05)
        decay = TemporalDecay(config)
        result = decay.compute(importance=1.0, age_days=365)
        assert result.decayed == 0.05

    def test_negative_age_treated_as_zero(self) -> None:
        decay = TemporalDecay()
        result = decay.compute(importance=0.5, age_days=-5)
        assert result.decayed == 0.5


class TestTemporalDecayReinforcement:
    def test_reinforcement_boosts_importance(self) -> None:
        decay = TemporalDecay()
        base = decay.compute(importance=0.5, age_days=0)
        boosted = decay.compute(importance=0.5, age_days=0, reinforcement_count=3)
        assert boosted.decayed > base.decayed

    def test_reinforcement_capped_at_max(self) -> None:
        config = DecayConfig(max_importance=1.0, reinforcement_boost=0.5)
        decay = TemporalDecay(config)
        result = decay.compute(importance=0.8, age_days=0, reinforcement_count=10)
        assert result.decayed <= 1.0

    def test_reinforcement_resists_decay(self) -> None:
        decay = TemporalDecay()
        without = decay.compute(importance=0.5, age_days=30)
        with_reinforcement = decay.compute(
            importance=0.5, age_days=30, reinforcement_count=5
        )
        assert with_reinforcement.decayed > without.decayed


class TestTemporalDecayAge:
    def test_compute_age_days(self) -> None:
        decay = TemporalDecay()
        now = datetime(2026, 4, 5, tzinfo=timezone.utc)
        created = datetime(2026, 4, 3, tzinfo=timezone.utc)
        age = decay.compute_age_days(created, now=now)
        assert abs(age - 2.0) < 0.001

    def test_compute_age_fractional(self) -> None:
        decay = TemporalDecay()
        now = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
        created = datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc)
        age = decay.compute_age_days(created, now=now)
        assert abs(age - 0.5) < 0.001

    def test_compute_age_naive_datetimes(self) -> None:
        decay = TemporalDecay()
        now = datetime(2026, 4, 5)
        created = datetime(2026, 4, 1)
        age = decay.compute_age_days(created, now=now)
        assert abs(age - 4.0) < 0.001


class TestTemporalDecayBatch:
    def test_batch_compute(self) -> None:
        decay = TemporalDecay()
        items = [
            {"importance": 0.9, "age_days": 0},
            {"importance": 0.5, "age_days": 30},
            {"importance": 0.3, "age_days": 60},
        ]
        results = decay.batch_compute(items)
        assert len(results) == 3
        assert results[0].decayed == 0.9  # no decay
        assert results[1].decayed < 0.5   # decayed
        assert results[2].decayed < results[1].decayed  # more decay

    def test_batch_with_reinforcement(self) -> None:
        decay = TemporalDecay()
        items = [
            {"importance": 0.5, "age_days": 30, "reinforcement_count": 5},
            {"importance": 0.5, "age_days": 30, "reinforcement_count": 0},
        ]
        results = decay.batch_compute(items)
        assert results[0].decayed > results[1].decayed

    def test_batch_empty(self) -> None:
        decay = TemporalDecay()
        results = decay.batch_compute([])
        assert results == []
