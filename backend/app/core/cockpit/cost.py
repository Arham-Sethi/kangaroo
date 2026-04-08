"""Cost tracking — real-time token counting and cost calculation per model.

Pricing is based on per-1M-token rates for each model. Costs accumulate
per cockpit session and per user for analytics.

Usage:
    tracker = CostTracker()
    cost = tracker.calculate("openai/gpt-4o", prompt_tokens=1000, completion_tokens=500)
    tracker.record(session_id, "openai/gpt-4o", prompt_tokens=1000, completion_tokens=500)
    total = tracker.get_session_total(session_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a specific model.

    Attributes:
        model_id: Model identifier (e.g., "openai/gpt-4o").
        prompt_cost_per_1m: Cost per 1M prompt tokens in USD.
        completion_cost_per_1m: Cost per 1M completion tokens in USD.
        display_name: Human-readable model name.
    """

    model_id: str
    prompt_cost_per_1m: float
    completion_cost_per_1m: float
    display_name: str = ""


@dataclass(frozen=True)
class CostEntry:
    """A single cost record.

    Attributes:
        model_id: Model that generated the cost.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.
        cost_usd: Total cost in USD.
    """

    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class SessionCostSummary:
    """Aggregated cost summary for a session.

    Attributes:
        session_id: The session identifier.
        total_prompt_tokens: Total prompt tokens across all models.
        total_completion_tokens: Total completion tokens.
        total_cost_usd: Total cost in USD.
        entries: Individual cost entries.
        cost_by_model: Cost breakdown per model.
    """

    session_id: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    entries: tuple[CostEntry, ...]
    cost_by_model: dict[str, float]


# Default pricing (approximate, as of early 2025)
_DEFAULT_PRICING: dict[str, ModelPricing] = {
    "openai/gpt-4o": ModelPricing(
        model_id="openai/gpt-4o",
        prompt_cost_per_1m=2.50,
        completion_cost_per_1m=10.00,
        display_name="GPT-4o",
    ),
    "openai/gpt-4o-mini": ModelPricing(
        model_id="openai/gpt-4o-mini",
        prompt_cost_per_1m=0.15,
        completion_cost_per_1m=0.60,
        display_name="GPT-4o Mini",
    ),
    "anthropic/claude-sonnet-4": ModelPricing(
        model_id="anthropic/claude-sonnet-4",
        prompt_cost_per_1m=3.00,
        completion_cost_per_1m=15.00,
        display_name="Claude Sonnet 4",
    ),
    "anthropic/claude-haiku-4": ModelPricing(
        model_id="anthropic/claude-haiku-4",
        prompt_cost_per_1m=0.80,
        completion_cost_per_1m=4.00,
        display_name="Claude Haiku 4",
    ),
    "google/gemini-2.0-flash": ModelPricing(
        model_id="google/gemini-2.0-flash",
        prompt_cost_per_1m=0.10,
        completion_cost_per_1m=0.40,
        display_name="Gemini 2.0 Flash",
    ),
    "google/gemini-2.5-pro": ModelPricing(
        model_id="google/gemini-2.5-pro",
        prompt_cost_per_1m=1.25,
        completion_cost_per_1m=10.00,
        display_name="Gemini 2.5 Pro",
    ),
}


class CostTracker:
    """Tracks token usage and costs across cockpit sessions.

    Maintains an in-memory ledger per session. In production,
    this would be backed by the database.
    """

    def __init__(
        self,
        custom_pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._pricing = dict(_DEFAULT_PRICING)
        if custom_pricing:
            self._pricing.update(custom_pricing)
        # session_id -> list of CostEntry
        self._ledger: dict[str, list[CostEntry]] = {}

    @property
    def known_models(self) -> list[str]:
        """List all models with known pricing."""
        return sorted(self._pricing.keys())

    def get_pricing(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a model."""
        return self._pricing.get(model_id)

    def calculate(
        self,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> float:
        """Calculate cost for a given token usage.

        Returns cost in USD. Unknown models return 0.0.
        """
        pricing = self._pricing.get(model_id)
        if pricing is None:
            return 0.0

        prompt_cost = (prompt_tokens / 1_000_000) * pricing.prompt_cost_per_1m
        completion_cost = (
            completion_tokens / 1_000_000
        ) * pricing.completion_cost_per_1m

        return prompt_cost + completion_cost

    def record(
        self,
        session_id: str,
        model_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> CostEntry:
        """Record a cost entry for a session.

        Returns the created CostEntry.
        """
        cost = self.calculate(model_id, prompt_tokens, completion_tokens)
        entry = CostEntry(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

        if session_id not in self._ledger:
            self._ledger[session_id] = []
        self._ledger[session_id].append(entry)

        return entry

    def get_session_total(self, session_id: str) -> float:
        """Get total cost for a session in USD."""
        entries = self._ledger.get(session_id, [])
        return sum(e.cost_usd for e in entries)

    def get_session_summary(self, session_id: str) -> SessionCostSummary:
        """Get detailed cost summary for a session."""
        entries = self._ledger.get(session_id, [])

        total_prompt = sum(e.prompt_tokens for e in entries)
        total_completion = sum(e.completion_tokens for e in entries)
        total_cost = sum(e.cost_usd for e in entries)

        cost_by_model: dict[str, float] = {}
        for entry in entries:
            cost_by_model[entry.model_id] = (
                cost_by_model.get(entry.model_id, 0.0) + entry.cost_usd
            )

        return SessionCostSummary(
            session_id=session_id,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cost_usd=total_cost,
            entries=tuple(entries),
            cost_by_model=cost_by_model,
        )

    def clear_session(self, session_id: str) -> None:
        """Clear all cost records for a session."""
        self._ledger.pop(session_id, None)
