"""Tests for ConsensusEngine — parallel query and response comparison.

Tests cover:
    - Agreement detection (strong, partial, disagreement)
    - Common theme extraction
    - Difference detection
    - Error handling
    - Empty/insufficient inputs
    - Pairwise overlap calculation
"""

import pytest

from app.core.cockpit.consensus import (
    AgreementLevel,
    ConsensusEngine,
    ConsensusResult,
    _compute_pairwise_overlap,
    _extract_key_phrases,
    _find_common_themes,
    _find_differences,
)
from app.core.cockpit.orchestrator import ModelResponse


# -- Mock calls --------------------------------------------------------------


async def _agreeing_call(model_id: str, prompt: str, ctx: str) -> ModelResponse:
    """All models give similar responses."""
    return ModelResponse(
        model_id=model_id,
        content="Python is excellent for backend development with FastAPI framework and PostgreSQL database",
        prompt_tokens=50,
        completion_tokens=20,
    )


async def _disagreeing_call(model_id: str, prompt: str, ctx: str) -> ModelResponse:
    """Each model gives very different responses."""
    responses = {
        "m1": "Python Flask microservices architecture serverless",
        "m2": "Java Spring monolith traditional enterprise deployment",
        "m3": "Rust Actix performance systems programming embedded",
    }
    return ModelResponse(
        model_id=model_id,
        content=responses.get(model_id, "Unknown response content here"),
        prompt_tokens=30,
        completion_tokens=15,
    )


async def _partial_error_call(model_id: str, prompt: str, ctx: str) -> ModelResponse:
    if model_id == "bad":
        raise RuntimeError("API failure")
    return ModelResponse(
        model_id=model_id,
        content="Valid response about Python development",
        prompt_tokens=30,
        completion_tokens=15,
    )


# -- Helper function tests ---------------------------------------------------


class TestExtractKeyPhrases:
    def test_basic(self) -> None:
        phrases = _extract_key_phrases("Python is great for web development")
        assert "python" in phrases
        assert "great" in phrases
        assert "development" in phrases

    def test_filters_stop_words(self) -> None:
        phrases = _extract_key_phrases("the and for are but not")
        assert len(phrases) == 0

    def test_filters_short_words(self) -> None:
        phrases = _extract_key_phrases("go is ok")
        assert len(phrases) == 0  # All <= 2 chars


class TestPairwiseOverlap:
    def test_identical_texts(self) -> None:
        overlap = _compute_pairwise_overlap([
            "Python FastAPI REST API backend",
            "Python FastAPI REST API backend",
        ])
        assert overlap == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        overlap = _compute_pairwise_overlap([
            "Python FastAPI backend development",
            "Kubernetes Docker containerization orchestration",
        ])
        assert overlap < 0.3

    def test_single_text(self) -> None:
        assert _compute_pairwise_overlap(["single"]) == 1.0

    def test_empty(self) -> None:
        assert _compute_pairwise_overlap([]) == 1.0


class TestFindCommonThemes:
    def test_finds_common(self) -> None:
        texts = [
            "Python backend framework development",
            "Python backend REST API development",
        ]
        common = _find_common_themes(texts, min_count=2)
        assert "python" in common
        assert "backend" in common
        assert "development" in common

    def test_no_common(self) -> None:
        texts = [
            "Python backend framework",
            "Kubernetes Docker containers",
        ]
        common = _find_common_themes(texts, min_count=2)
        # No overlap expected
        assert len(common) == 0

    def test_single_text(self) -> None:
        assert _find_common_themes(["solo"]) == []


class TestFindDifferences:
    def test_finds_unique(self) -> None:
        texts = [
            "Python FastAPI backend async development",
            "Python Django backend synchronous development",
        ]
        diffs = _find_differences(texts)
        assert "fastapi" in diffs or "django" in diffs


# -- ConsensusEngine tests ---------------------------------------------------


class TestConsensusEngine:
    @pytest.mark.asyncio
    async def test_strong_agreement(self) -> None:
        engine = ConsensusEngine(call_fn=_agreeing_call)
        result = await engine.evaluate(
            prompt="What's best for backend?",
            models=["m1", "m2", "m3"],
        )
        assert isinstance(result, ConsensusResult)
        assert result.agreement_level == AgreementLevel.STRONG
        assert result.agreement_score > 0.3
        assert len(result.common_themes) > 0

    @pytest.mark.asyncio
    async def test_disagreement(self) -> None:
        engine = ConsensusEngine(call_fn=_disagreeing_call)
        result = await engine.evaluate(
            prompt="What's best?",
            models=["m1", "m2", "m3"],
        )
        assert result.agreement_level in (
            AgreementLevel.PARTIAL,
            AgreementLevel.DISAGREEMENT,
        )
        assert result.agreement_score < 0.5

    @pytest.mark.asyncio
    async def test_empty_models(self) -> None:
        engine = ConsensusEngine()
        result = await engine.evaluate(prompt="Test", models=[])
        assert result.agreement_level == AgreementLevel.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_single_model_insufficient(self) -> None:
        engine = ConsensusEngine(call_fn=_agreeing_call)
        result = await engine.evaluate(prompt="Test", models=["m1"])
        assert result.agreement_level == AgreementLevel.INSUFFICIENT

    @pytest.mark.asyncio
    async def test_error_handling(self) -> None:
        engine = ConsensusEngine(call_fn=_partial_error_call)
        result = await engine.evaluate(
            prompt="Test", models=["good1", "bad", "good2"]
        )
        assert "bad" in result.responses
        assert result.responses["bad"].is_error
        # Should still have enough valid responses
        assert len(result.responses) == 3

    @pytest.mark.asyncio
    async def test_summary_present(self) -> None:
        engine = ConsensusEngine(call_fn=_agreeing_call)
        result = await engine.evaluate(
            prompt="Test", models=["m1", "m2"]
        )
        assert result.summary
        assert "Agreement" in result.summary

    @pytest.mark.asyncio
    async def test_latency_tracked(self) -> None:
        engine = ConsensusEngine(call_fn=_agreeing_call)
        result = await engine.evaluate(
            prompt="Test", models=["m1", "m2"]
        )
        assert result.total_latency_ms >= 0
