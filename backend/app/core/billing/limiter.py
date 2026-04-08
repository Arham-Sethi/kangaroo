"""Rate limiter — token bucket algorithm for API request throttling.

Implements a per-user token bucket that refills at a steady rate.
Each API request consumes one token. When the bucket is empty,
requests are rejected with 429 Too Many Requests.

Token bucket parameters:
    - capacity: Maximum tokens (burst size + hourly rate)
    - refill_rate: Tokens added per second (requests_per_hour / 3600)

Production: Redis-backed for horizontal scaling.
Development: In-memory dict.

Usage:
    limiter = RateLimiter()
    result = limiter.check(user_id, tier_config.rate_limit)
    if not result.allowed:
        raise HTTPException(429, detail=result.to_dict())
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.billing.tiers import RateLimitConfig, UNLIMITED


@dataclass(frozen=True)
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed.
        remaining: Tokens remaining in the bucket.
        limit: Maximum tokens (capacity).
        reset_in_seconds: Seconds until one token is available.
        retry_after: Seconds to wait before retrying (0 if allowed).
    """

    allowed: bool
    remaining: int
    limit: int
    reset_in_seconds: float
    retry_after: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "limit": self.limit,
            "reset_in_seconds": round(self.reset_in_seconds, 1),
            "retry_after": round(self.retry_after, 1),
        }

    def to_headers(self) -> dict[str, str]:
        """Generate standard rate-limit HTTP headers."""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(round(self.reset_in_seconds)),
            **({"Retry-After": str(round(self.retry_after))} if not self.allowed else {}),
        }


class _TokenBucket:
    """Single user's token bucket state."""

    __slots__ = ("tokens", "last_refill", "capacity", "refill_rate")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.tokens: float = float(capacity)
        self.last_refill: float = time.monotonic()
        self.capacity: int = capacity
        self.refill_rate: float = refill_rate

    def refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now

    def consume(self) -> bool:
        """Try to consume one token. Returns True if successful."""
        self.refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def time_until_available(self) -> float:
        """Seconds until one token is available."""
        if self.tokens >= 1.0:
            return 0.0
        if self.refill_rate <= 0:
            return float("inf")
        deficit = 1.0 - self.tokens
        return deficit / self.refill_rate


class RateLimiter:
    """Per-user rate limiter using token bucket algorithm.

    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _TokenBucket] = {}

    def check(
        self,
        user_id: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit and consume a token if allowed.

        Args:
            user_id: The user making the request.
            config: Rate limit configuration for the user's tier.

        Returns:
            RateLimitResult indicating whether the request is allowed.
        """
        # Unlimited tier — always allow
        if config.requests_per_hour == UNLIMITED:
            return RateLimitResult(
                allowed=True,
                remaining=UNLIMITED,
                limit=UNLIMITED,
                reset_in_seconds=0.0,
                retry_after=0.0,
            )

        bucket = self._get_or_create_bucket(user_id, config)

        if bucket.consume():
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket.tokens),
                limit=bucket.capacity,
                reset_in_seconds=0.0,
                retry_after=0.0,
            )

        wait_time = bucket.time_until_available()
        return RateLimitResult(
            allowed=False,
            remaining=0,
            limit=bucket.capacity,
            reset_in_seconds=wait_time,
            retry_after=wait_time,
        )

    def peek(
        self,
        user_id: str,
        config: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limit without consuming a token."""
        if config.requests_per_hour == UNLIMITED:
            return RateLimitResult(
                allowed=True,
                remaining=UNLIMITED,
                limit=UNLIMITED,
                reset_in_seconds=0.0,
                retry_after=0.0,
            )

        bucket = self._get_or_create_bucket(user_id, config)
        bucket.refill()

        allowed = bucket.tokens >= 1.0
        return RateLimitResult(
            allowed=allowed,
            remaining=int(bucket.tokens),
            limit=bucket.capacity,
            reset_in_seconds=0.0 if allowed else bucket.time_until_available(),
            retry_after=0.0 if allowed else bucket.time_until_available(),
        )

    def _get_or_create_bucket(
        self, user_id: str, config: RateLimitConfig
    ) -> _TokenBucket:
        """Get or create a token bucket for a user."""
        capacity = config.requests_per_hour + config.burst_size
        refill_rate = config.requests_per_hour / 3600.0

        bucket = self._buckets.get(user_id)
        if bucket is None or bucket.capacity != capacity:
            bucket = _TokenBucket(capacity=capacity, refill_rate=refill_rate)
            self._buckets[user_id] = bucket

        return bucket

    def reset(self, user_id: str) -> None:
        """Reset rate limit for a user (refill bucket)."""
        self._buckets.pop(user_id, None)

    def reset_all(self) -> None:
        """Reset all rate limits."""
        self._buckets.clear()

    @property
    def tracked_users(self) -> int:
        """Number of users being tracked."""
        return len(self._buckets)
