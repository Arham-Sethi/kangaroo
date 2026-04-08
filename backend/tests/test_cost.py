"""Tests for CostTracker — token counting and cost calculation.

Tests cover:
    - Cost calculation per model
    - Recording cost entries
    - Session totals
    - Session summaries
    - Cost by model breakdown
    - Unknown models
    - Custom pricing
    - Clear session
"""

import pytest

from app.core.cockpit.cost import (
    CostEntry,
    CostTracker,
    ModelPricing,
    SessionCostSummary,
)


class TestCostCalculation:
    def test_known_model(self) -> None:
        tracker = CostTracker()
        # GPT-4o: $2.50/1M prompt, $10/1M completion
        cost = tracker.calculate(
            "openai/gpt-4o", prompt_tokens=1_000_000, completion_tokens=0
        )
        assert cost == pytest.approx(2.50, abs=0.01)

    def test_completion_cost(self) -> None:
        tracker = CostTracker()
        cost = tracker.calculate(
            "openai/gpt-4o", prompt_tokens=0, completion_tokens=1_000_000
        )
        assert cost == pytest.approx(10.00, abs=0.01)

    def test_combined_cost(self) -> None:
        tracker = CostTracker()
        cost = tracker.calculate(
            "openai/gpt-4o", prompt_tokens=500_000, completion_tokens=200_000
        )
        expected = (500_000 / 1_000_000) * 2.50 + (200_000 / 1_000_000) * 10.00
        assert cost == pytest.approx(expected, abs=0.001)

    def test_unknown_model_returns_zero(self) -> None:
        tracker = CostTracker()
        cost = tracker.calculate("unknown/model", prompt_tokens=1000)
        assert cost == 0.0

    def test_zero_tokens(self) -> None:
        tracker = CostTracker()
        cost = tracker.calculate("openai/gpt-4o", prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0

    def test_custom_pricing(self) -> None:
        custom = {
            "custom/model": ModelPricing(
                model_id="custom/model",
                prompt_cost_per_1m=1.0,
                completion_cost_per_1m=2.0,
            )
        }
        tracker = CostTracker(custom_pricing=custom)
        cost = tracker.calculate("custom/model", prompt_tokens=1_000_000)
        assert cost == pytest.approx(1.0)


class TestCostRecording:
    def test_record_entry(self) -> None:
        tracker = CostTracker()
        entry = tracker.record(
            "sess1", "openai/gpt-4o", prompt_tokens=1000, completion_tokens=500
        )
        assert isinstance(entry, CostEntry)
        assert entry.model_id == "openai/gpt-4o"
        assert entry.prompt_tokens == 1000
        assert entry.cost_usd > 0

    def test_session_total(self) -> None:
        tracker = CostTracker()
        tracker.record("sess1", "openai/gpt-4o", prompt_tokens=1000)
        tracker.record("sess1", "openai/gpt-4o", prompt_tokens=2000)
        total = tracker.get_session_total("sess1")
        assert total > 0

    def test_empty_session_total(self) -> None:
        tracker = CostTracker()
        assert tracker.get_session_total("nonexistent") == 0.0

    def test_multiple_models(self) -> None:
        tracker = CostTracker()
        tracker.record("sess1", "openai/gpt-4o", prompt_tokens=1000)
        tracker.record("sess1", "anthropic/claude-sonnet-4", prompt_tokens=1000)
        total = tracker.get_session_total("sess1")
        assert total > 0


class TestSessionSummary:
    def test_summary(self) -> None:
        tracker = CostTracker()
        tracker.record("sess1", "openai/gpt-4o", prompt_tokens=1000, completion_tokens=500)
        tracker.record("sess1", "anthropic/claude-sonnet-4", prompt_tokens=2000, completion_tokens=800)

        summary = tracker.get_session_summary("sess1")
        assert isinstance(summary, SessionCostSummary)
        assert summary.session_id == "sess1"
        assert summary.total_prompt_tokens == 3000
        assert summary.total_completion_tokens == 1300
        assert summary.total_cost_usd > 0
        assert len(summary.entries) == 2
        assert "openai/gpt-4o" in summary.cost_by_model
        assert "anthropic/claude-sonnet-4" in summary.cost_by_model

    def test_empty_summary(self) -> None:
        tracker = CostTracker()
        summary = tracker.get_session_summary("empty")
        assert summary.total_cost_usd == 0.0
        assert summary.entries == ()

    def test_clear_session(self) -> None:
        tracker = CostTracker()
        tracker.record("sess1", "openai/gpt-4o", prompt_tokens=1000)
        tracker.clear_session("sess1")
        assert tracker.get_session_total("sess1") == 0.0


class TestKnownModels:
    def test_list_known_models(self) -> None:
        tracker = CostTracker()
        models = tracker.known_models
        assert "openai/gpt-4o" in models
        assert "anthropic/claude-sonnet-4" in models
        assert "google/gemini-2.0-flash" in models

    def test_get_pricing(self) -> None:
        tracker = CostTracker()
        pricing = tracker.get_pricing("openai/gpt-4o")
        assert pricing is not None
        assert pricing.prompt_cost_per_1m > 0
        assert pricing.display_name == "GPT-4o"

    def test_get_pricing_unknown(self) -> None:
        tracker = CostTracker()
        assert tracker.get_pricing("unknown") is None
