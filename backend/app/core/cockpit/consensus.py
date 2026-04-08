"""Consensus engine — parallel query and response comparison.

Sends the same prompt to multiple models, collects responses,
and synthesizes agreement/disagreement.

Useful for:
    - Fact-checking (do all models agree?)
    - Getting diverse perspectives
    - Building confidence in an answer

Usage:
    engine = ConsensusEngine(call_fn=my_llm_call)
    result = await engine.evaluate(
        prompt="What's the best database for this use case?",
        models=["openai/gpt-4o", "anthropic/claude-sonnet-4", "google/gemini-2.5-pro"],
    )
    print(result.agreement_level)  # "strong" | "partial" | "disagreement"
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.core.cockpit.orchestrator import ModelCallFn, ModelResponse


class AgreementLevel:
    """Constants for agreement levels."""

    STRONG = "strong"          # All models substantially agree
    PARTIAL = "partial"        # Some agreement, some differences
    DISAGREEMENT = "disagreement"  # Models significantly disagree
    INSUFFICIENT = "insufficient"  # Not enough valid responses


@dataclass(frozen=True)
class ConsensusResult:
    """Result of a consensus evaluation.

    Attributes:
        prompt: The original prompt.
        responses: Map of model_id -> ModelResponse.
        agreement_level: How much the models agree.
        agreement_score: Numeric agreement score (0..1).
        common_themes: Themes found across multiple responses.
        differences: Notable differences between responses.
        summary: Human-readable consensus summary.
        total_latency_ms: Wall-clock time for all responses.
    """

    prompt: str
    responses: dict[str, ModelResponse]
    agreement_level: str
    agreement_score: float
    common_themes: tuple[str, ...]
    differences: tuple[str, ...]
    summary: str
    total_latency_ms: float


def _extract_key_phrases(text: str) -> set[str]:
    """Extract significant phrases from a response for comparison."""
    # Simple tokenization — production would use NLP
    tokens = set(re.findall(r"[a-z]{3,}", text.lower()))
    # Filter out common stop words
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "with", "that", "this",
        "will", "each", "make", "like", "been", "have", "from", "they",
        "some", "than", "them", "then", "also", "into", "just", "more",
        "such", "when", "very", "what", "which", "would", "about", "could",
        "other", "there", "their", "these", "should",
    }
    return tokens - stop_words


def _compute_pairwise_overlap(texts: list[str]) -> float:
    """Compute average pairwise Jaccard overlap of key phrases."""
    if len(texts) < 2:
        return 1.0

    phrase_sets = [_extract_key_phrases(t) for t in texts]
    overlaps: list[float] = []

    for i in range(len(phrase_sets)):
        for j in range(i + 1, len(phrase_sets)):
            a, b = phrase_sets[i], phrase_sets[j]
            if not a or not b:
                overlaps.append(0.0)
                continue
            intersection = len(a & b)
            union = len(a | b)
            overlaps.append(intersection / union if union else 0.0)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _find_common_themes(texts: list[str], min_count: int = 2) -> list[str]:
    """Find phrases that appear in multiple responses."""
    if len(texts) < 2:
        return []

    phrase_sets = [_extract_key_phrases(t) for t in texts]

    # Count how many responses contain each phrase
    counts: dict[str, int] = {}
    for phrases in phrase_sets:
        for phrase in phrases:
            counts[phrase] = counts.get(phrase, 0) + 1

    common = [
        phrase for phrase, count in counts.items()
        if count >= min_count and len(phrase) > 3
    ]
    return sorted(common, key=lambda p: -counts[p])[:10]


def _find_differences(texts: list[str]) -> list[str]:
    """Find phrases unique to individual responses."""
    if len(texts) < 2:
        return []

    phrase_sets = [_extract_key_phrases(t) for t in texts]
    all_phrases = set()
    for ps in phrase_sets:
        all_phrases |= ps

    unique: list[str] = []
    for phrases in phrase_sets:
        for phrase in phrases:
            # Phrase appears in only this response
            count = sum(1 for ps in phrase_sets if phrase in ps)
            if count == 1 and len(phrase) > 4:
                unique.append(phrase)

    return sorted(set(unique))[:10]


class ConsensusEngine:
    """Evaluates consensus across multiple model responses.

    Sends the same prompt to all models in parallel, then analyzes
    the responses for agreement/disagreement.
    """

    def __init__(self, call_fn: ModelCallFn | None = None) -> None:
        self._call_fn = call_fn or _default_consensus_call

    async def evaluate(
        self,
        prompt: str,
        models: list[str],
        system_context: str = "",
    ) -> ConsensusResult:
        """Evaluate consensus across models.

        Args:
            prompt: The prompt to send to all models.
            models: List of model identifiers.
            system_context: Shared system context.

        Returns:
            ConsensusResult with agreement analysis.
        """
        if not models:
            return ConsensusResult(
                prompt=prompt,
                responses={},
                agreement_level=AgreementLevel.INSUFFICIENT,
                agreement_score=0.0,
                common_themes=(),
                differences=(),
                summary="No models provided.",
                total_latency_ms=0.0,
            )

        start = time.monotonic()

        # Dispatch in parallel
        tasks = [
            asyncio.create_task(self._call_fn(model_id, prompt, system_context))
            for model_id in models
        ]
        responses_list = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Build response map
        responses: dict[str, ModelResponse] = {}
        for model_id, result in zip(models, responses_list):
            if isinstance(result, Exception):
                responses[model_id] = ModelResponse(
                    model_id=model_id, error=str(result)
                )
            else:
                responses[model_id] = result

        # Filter successful responses for analysis
        valid_texts = [
            r.content for r in responses.values()
            if not r.is_error and r.content.strip()
        ]

        if len(valid_texts) < 2:
            return ConsensusResult(
                prompt=prompt,
                responses=responses,
                agreement_level=AgreementLevel.INSUFFICIENT,
                agreement_score=0.0,
                common_themes=(),
                differences=(),
                summary="Not enough valid responses for consensus analysis.",
                total_latency_ms=round(elapsed_ms, 2),
            )

        # Analyze agreement
        overlap_score = _compute_pairwise_overlap(valid_texts)
        common = _find_common_themes(valid_texts)
        diffs = _find_differences(valid_texts)

        if overlap_score >= 0.4:
            level = AgreementLevel.STRONG
        elif overlap_score >= 0.2:
            level = AgreementLevel.PARTIAL
        else:
            level = AgreementLevel.DISAGREEMENT

        summary = (
            f"Agreement: {level} ({overlap_score:.0%} overlap). "
            f"{len(common)} common themes, {len(diffs)} unique points."
        )

        return ConsensusResult(
            prompt=prompt,
            responses=responses,
            agreement_level=level,
            agreement_score=round(overlap_score, 4),
            common_themes=tuple(common),
            differences=tuple(diffs),
            summary=summary,
            total_latency_ms=round(elapsed_ms, 2),
        )


async def _default_consensus_call(
    model_id: str, prompt: str, system_context: str
) -> ModelResponse:
    """Default consensus call — routes to the real LLM client."""
    from app.core.llm.client import LLMClient
    from app.core.llm.models import LLMError, LLMMessage

    client = LLMClient()
    messages: list[LLMMessage] = []

    if system_context:
        messages.append(LLMMessage(role="system", content=system_context))
    messages.append(LLMMessage(role="user", content=prompt))

    try:
        response = await client.call(model_id, messages)
        return ModelResponse(
            model_id=model_id,
            content=response.content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
    except LLMError as exc:
        return ModelResponse(
            model_id=model_id,
            error=str(exc),
        )
