"""LLM data models — immutable types for requests, responses, and configuration.

All dataclasses are frozen (immutable) to prevent accidental mutation and
enable safe sharing across async tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ── Provider Enum ────────────────────────────────────────────────────────────


class ModelProvider(StrEnum):
    """Supported LLM API providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# ── Message Format ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMMessage:
    """A single message in a conversation.

    This is the unified format — the client converts it to provider-specific
    formats (OpenAI uses role/content, Anthropic separates system, etc.).
    """

    role: str  # "system", "user", "assistant"
    content: str


# ── Response Types ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMResponse:
    """Complete response from an LLM call.

    Attributes:
        content: The generated text.
        model: Actual model used (may differ from requested, e.g., fallback).
        provider: Which provider served this response.
        prompt_tokens: Input token count.
        completion_tokens: Output token count.
        latency_ms: Round-trip time in milliseconds.
        cost_usd: Estimated cost based on token pricing.
        metadata: Provider-specific extras (finish_reason, etc.).
    """

    content: str
    model: str
    provider: ModelProvider
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class LLMChunk:
    """A single streaming chunk from an LLM.

    Attributes:
        delta: The text fragment in this chunk.
        model: Model producing this chunk.
        is_final: True if this is the last chunk (may contain token counts).
        prompt_tokens: Populated only on the final chunk.
        completion_tokens: Populated only on the final chunk.
    """

    delta: str
    model: str
    is_final: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ── Error Type ───────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Unified error for all LLM API failures.

    Wraps provider-specific errors into a single type so callers don't need
    to handle httpx, openai, anthropic, and google errors separately.

    Attributes:
        message: Human-readable error description.
        provider: Which provider failed.
        model: Which model was being called.
        status_code: HTTP status code (if applicable).
        is_retryable: Whether the caller should retry.
        original_error: The underlying exception.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        is_retryable: bool = False,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.is_retryable = is_retryable
        self.original_error = original_error


# ── Model Registry ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a supported model.

    Attributes:
        model_id: Canonical ID (e.g., "gpt-4o").
        provider: Which provider serves this model.
        api_model_name: Exact string to send to the provider API.
        prompt_cost_per_1m: Cost per 1M prompt tokens in USD.
        completion_cost_per_1m: Cost per 1M completion tokens in USD.
        max_context: Maximum context window in tokens.
        display_name: Human-friendly name for UI.
    """

    model_id: str
    provider: ModelProvider
    api_model_name: str
    prompt_cost_per_1m: float
    completion_cost_per_1m: float
    max_context: int
    display_name: str

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for given token counts."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.prompt_cost_per_1m
        completion_cost = (completion_tokens / 1_000_000) * self.completion_cost_per_1m
        return round(prompt_cost + completion_cost, 6)


# Registry of all supported models with pricing and metadata.
# Pricing data as of April 2026.
MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-4o": ModelInfo(
        model_id="gpt-4o",
        provider=ModelProvider.OPENAI,
        api_model_name="gpt-4o",
        prompt_cost_per_1m=2.50,
        completion_cost_per_1m=10.00,
        max_context=128_000,
        display_name="GPT-4o",
    ),
    "gpt-4o-mini": ModelInfo(
        model_id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        api_model_name="gpt-4o-mini",
        prompt_cost_per_1m=0.15,
        completion_cost_per_1m=0.60,
        max_context=128_000,
        display_name="GPT-4o Mini",
    ),
    "gpt-4.1": ModelInfo(
        model_id="gpt-4.1",
        provider=ModelProvider.OPENAI,
        api_model_name="gpt-4.1",
        prompt_cost_per_1m=2.00,
        completion_cost_per_1m=8.00,
        max_context=1_000_000,
        display_name="GPT-4.1",
    ),
    "gpt-4.1-mini": ModelInfo(
        model_id="gpt-4.1-mini",
        provider=ModelProvider.OPENAI,
        api_model_name="gpt-4.1-mini",
        prompt_cost_per_1m=0.40,
        completion_cost_per_1m=1.60,
        max_context=1_000_000,
        display_name="GPT-4.1 Mini",
    ),
    # ── Anthropic ────────────────────────────────────────────────────────
    "claude-sonnet-4": ModelInfo(
        model_id="claude-sonnet-4",
        provider=ModelProvider.ANTHROPIC,
        api_model_name="claude-sonnet-4-20250514",
        prompt_cost_per_1m=3.00,
        completion_cost_per_1m=15.00,
        max_context=200_000,
        display_name="Claude Sonnet 4",
    ),
    "claude-haiku-4": ModelInfo(
        model_id="claude-haiku-4",
        provider=ModelProvider.ANTHROPIC,
        api_model_name="claude-haiku-4-20250514",
        prompt_cost_per_1m=0.80,
        completion_cost_per_1m=4.00,
        max_context=200_000,
        display_name="Claude Haiku 4",
    ),
    "claude-opus-4": ModelInfo(
        model_id="claude-opus-4",
        provider=ModelProvider.ANTHROPIC,
        api_model_name="claude-opus-4-20250514",
        prompt_cost_per_1m=15.00,
        completion_cost_per_1m=75.00,
        max_context=200_000,
        display_name="Claude Opus 4",
    ),
    # ── Google ───────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelInfo(
        model_id="gemini-2.0-flash",
        provider=ModelProvider.GOOGLE,
        api_model_name="gemini-2.0-flash",
        prompt_cost_per_1m=0.10,
        completion_cost_per_1m=0.40,
        max_context=1_000_000,
        display_name="Gemini 2.0 Flash",
    ),
    "gemini-2.5-pro": ModelInfo(
        model_id="gemini-2.5-pro",
        provider=ModelProvider.GOOGLE,
        api_model_name="gemini-2.5-pro-preview-05-06",
        prompt_cost_per_1m=1.25,
        completion_cost_per_1m=10.00,
        max_context=1_000_000,
        display_name="Gemini 2.5 Pro",
    ),
}


def resolve_model(model_id: str) -> ModelInfo:
    """Look up a model by ID, raising LLMError if not found."""
    info = MODEL_REGISTRY.get(model_id)
    if info is None:
        raise LLMError(
            f"Unknown model: {model_id}. "
            f"Supported: {', '.join(sorted(MODEL_REGISTRY.keys()))}",
            model=model_id,
        )
    return info
