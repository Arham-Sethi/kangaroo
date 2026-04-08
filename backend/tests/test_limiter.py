"""Tests for RateLimiter — token bucket rate limiting.

Tests cover:
    - _TokenBucket refill, consume, time_until_available
    - RateLimitResult to_dict() and to_headers()
    - RateLimiter.check() allowed and denied
    - RateLimiter.check() with unlimited config
    - RateLimiter.peek() without consuming
    - RateLimiter.reset() and reset_all()
    - Bucket recreation on config change
    - Multiple users isolation
    - tracked_users property
"""

import time

import pytest

from app.core.billing.tiers import UNLIMITED, RateLimitConfig
from app.core.billing.limiter import (
    RateLimitResult,
    RateLimiter,
    _TokenBucket,
)


class TestTokenBucket:
    """Test the internal _TokenBucket class."""

    def test_initial_tokens_equal_capacity(self) -> None:
        bucket = _TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10.0
        assert bucket.capacity == 10

    def test_consume_decreases_tokens(self) -> None:
        bucket = _TokenBucket(capacity=10, refill_rate=0.0)
        assert bucket.consume() is True
        assert bucket.tokens == 9.0

    def test_consume_when_empty(self) -> None:
        bucket = _TokenBucket(capacity=1, refill_rate=0.0)
        assert bucket.consume() is True  # Last token
        assert bucket.consume() is False  # Empty

    def test_time_until_available_when_full(self) -> None:
        bucket = _TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.time_until_available() == 0.0

    def test_time_until_available_when_empty(self) -> None:
        bucket = _TokenBucket(capacity=1, refill_rate=1.0)
        bucket.consume()
        wait = bucket.time_until_available()
        assert wait > 0.0

    def test_time_until_available_zero_refill(self) -> None:
        bucket = _TokenBucket(capacity=1, refill_rate=0.0)
        bucket.consume()
        assert bucket.time_until_available() == float("inf")

    def test_refill_adds_tokens(self) -> None:
        bucket = _TokenBucket(capacity=100, refill_rate=1000.0)  # Fast refill
        bucket.tokens = 0.0
        bucket.last_refill = time.monotonic() - 0.1  # 100ms ago
        bucket.refill()
        # ~100 tokens refilled in 0.1s at 1000/s
        assert bucket.tokens > 50.0

    def test_refill_capped_at_capacity(self) -> None:
        bucket = _TokenBucket(capacity=10, refill_rate=1000.0)
        bucket.last_refill = time.monotonic() - 1.0
        bucket.refill()
        assert bucket.tokens <= 10.0


class TestRateLimitResult:
    """Test RateLimitResult dataclass."""

    def test_to_dict_allowed(self) -> None:
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_in_seconds=0.0,
            retry_after=0.0,
        )
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["remaining"] == 5
        assert d["limit"] == 10

    def test_to_dict_denied(self) -> None:
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=10,
            reset_in_seconds=5.5,
            retry_after=5.5,
        )
        d = result.to_dict()
        assert d["allowed"] is False
        assert d["retry_after"] == 5.5

    def test_to_headers_allowed(self) -> None:
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            limit=10,
            reset_in_seconds=0.0,
            retry_after=0.0,
        )
        headers = result.to_headers()
        assert headers["X-RateLimit-Limit"] == "10"
        assert headers["X-RateLimit-Remaining"] == "5"
        assert headers["X-RateLimit-Reset"] == "0"
        assert "Retry-After" not in headers

    def test_to_headers_denied(self) -> None:
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            limit=10,
            reset_in_seconds=6.0,
            retry_after=6.0,
        )
        headers = result.to_headers()
        assert headers["Retry-After"] == "6"
        assert headers["X-RateLimit-Remaining"] == "0"

    def test_to_headers_negative_remaining_clamped(self) -> None:
        result = RateLimitResult(
            allowed=True,
            remaining=-1,
            limit=10,
            reset_in_seconds=0.0,
            retry_after=0.0,
        )
        headers = result.to_headers()
        assert headers["X-RateLimit-Remaining"] == "0"

    def test_frozen(self) -> None:
        result = RateLimitResult(
            allowed=True, remaining=5, limit=10,
            reset_in_seconds=0.0, retry_after=0.0,
        )
        with pytest.raises(AttributeError):
            result.allowed = False  # type: ignore[misc]


class TestRateLimiterCheck:
    """Test RateLimiter.check()."""

    def test_first_request_allowed(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=100, burst_size=10)
        result = limiter.check("user1", config)
        assert result.allowed is True
        assert result.limit == 110  # capacity = requests + burst

    def test_exhaust_bucket(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=5, burst_size=0)
        # Exhaust all 5 tokens
        for _ in range(5):
            result = limiter.check("user1", config)
            assert result.allowed is True
        # Next request denied
        result = limiter.check("user1", config)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after > 0

    def test_unlimited_always_allowed(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=UNLIMITED, burst_size=UNLIMITED)
        for _ in range(100):
            result = limiter.check("user1", config)
            assert result.allowed is True
        assert result.remaining == UNLIMITED
        assert result.limit == UNLIMITED

    def test_users_have_separate_buckets(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=2, burst_size=0)
        # User1 uses 2
        limiter.check("user1", config)
        limiter.check("user1", config)
        # User2 still has tokens
        result = limiter.check("user2", config)
        assert result.allowed is True


class TestRateLimiterPeek:
    """Test RateLimiter.peek() (read-only)."""

    def test_peek_does_not_consume(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=1, burst_size=0)
        peek1 = limiter.peek("user1", config)
        assert peek1.allowed is True
        peek2 = limiter.peek("user1", config)
        assert peek2.allowed is True  # Still allowed, not consumed

    def test_peek_reflects_consumption(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=1, burst_size=0)
        limiter.check("user1", config)  # Consume the one token
        peek = limiter.peek("user1", config)
        assert peek.allowed is False

    def test_peek_unlimited(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=UNLIMITED, burst_size=UNLIMITED)
        result = limiter.peek("user1", config)
        assert result.allowed is True
        assert result.remaining == UNLIMITED


class TestRateLimiterReset:
    """Test reset methods."""

    def test_reset_user(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=1, burst_size=0)
        limiter.check("user1", config)  # Use token
        limiter.reset("user1")
        result = limiter.check("user1", config)
        assert result.allowed is True  # Fresh bucket

    def test_reset_all(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=1, burst_size=0)
        limiter.check("user1", config)
        limiter.check("user2", config)
        limiter.reset_all()
        assert limiter.tracked_users == 0

    def test_reset_nonexistent_user(self) -> None:
        limiter = RateLimiter()
        limiter.reset("nobody")  # Should not raise


class TestRateLimiterTrackedUsers:
    """Test tracked_users property."""

    def test_initially_zero(self) -> None:
        limiter = RateLimiter()
        assert limiter.tracked_users == 0

    def test_increases_on_check(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=100, burst_size=10)
        limiter.check("user1", config)
        limiter.check("user2", config)
        assert limiter.tracked_users == 2

    def test_decreases_on_reset(self) -> None:
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_hour=100, burst_size=10)
        limiter.check("user1", config)
        limiter.reset("user1")
        assert limiter.tracked_users == 0


class TestRateLimiterBucketRecreation:
    """Test bucket recreation when config changes."""

    def test_new_bucket_on_capacity_change(self) -> None:
        limiter = RateLimiter()
        config1 = RateLimitConfig(requests_per_hour=10, burst_size=5)
        limiter.check("user1", config1)
        # Upgrade tier — new capacity
        config2 = RateLimitConfig(requests_per_hour=100, burst_size=50)
        result = limiter.check("user1", config2)
        assert result.allowed is True
        assert result.limit == 150  # New capacity
