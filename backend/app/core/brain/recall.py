"""Memory recall — hybrid search across all sessions.

Combines three ranking signals to find the most relevant past context:
    1. Keyword match — TF-IDF term overlap (fast, precise for exact terms)
    2. Semantic similarity — cosine similarity on embedding vectors
    3. Recency boost — exponential decay favors recent sessions

Final score = (w_keyword * keyword_score
             + w_semantic * semantic_score)
             * recency_factor

Usage:
    recall = MemoryRecall()
    results = recall.search(query="auth decisions", documents=docs)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecallConfig:
    """Weights and thresholds for hybrid recall.

    Attributes:
        keyword_weight: Weight for keyword match signal (0..1).
        semantic_weight: Weight for semantic similarity signal (0..1).
        recency_half_life_days: Half-life for recency decay.
        min_score: Minimum combined score to include in results.
        max_results: Maximum results to return.
    """

    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    recency_half_life_days: float = 30.0
    min_score: float = 0.01
    max_results: int = 20


@dataclass(frozen=True)
class RecallHit:
    """A recalled memory with composite score.

    Attributes:
        document_id: Identifier of the matched document.
        score: Final composite score (0..1).
        keyword_score: Raw keyword match score.
        semantic_score: Raw semantic similarity score.
        recency_factor: Recency decay multiplier applied.
        metadata: Additional metadata about the match.
    """

    document_id: str
    score: float
    keyword_score: float
    semantic_score: float
    recency_factor: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RecallDocument:
    """A document indexed for recall.

    Attributes:
        id: Unique document identifier.
        text: Original text content (for keyword matching).
        vector: Embedding vector (for semantic search).
        age_days: Age of the document in days (for recency).
        metadata: Additional metadata (session_id, entity names, etc.).
    """

    id: str
    text: str
    vector: tuple[float, ...] = ()
    age_days: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Compute keyword overlap score using Jaccard similarity.

    Returns a value between 0.0 and 1.0.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    intersection = len(query_tokens & doc_tokens)
    union = len(query_tokens | doc_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _recency_factor(age_days: float, half_life_days: float) -> float:
    """Compute recency decay factor.

    Returns 1.0 for age=0, 0.5 for age=half_life, etc.
    """
    if half_life_days <= 0 or age_days <= 0:
        return 1.0

    lam = math.log(2) / half_life_days
    return math.exp(-lam * age_days)


class MemoryRecall:
    """Hybrid search combining keyword, semantic, and recency signals.

    The recall engine scores each document against a query using three
    independent signals, then combines them into a single score:

        combined = (w_kw * keyword + w_sem * semantic) * recency

    Documents below min_score are filtered out. Results are sorted
    by combined score descending.
    """

    def __init__(self, config: RecallConfig | None = None) -> None:
        self._config = config or RecallConfig()

    @property
    def config(self) -> RecallConfig:
        return self._config

    def search(
        self,
        query: str,
        documents: list[RecallDocument],
        query_vector: tuple[float, ...] = (),
    ) -> list[RecallHit]:
        """Search documents using hybrid scoring.

        Args:
            query: The natural language query.
            documents: List of documents to search over.
            query_vector: Optional embedding vector for the query.
                         If empty, semantic scoring is disabled.

        Returns:
            List of RecallHit sorted by score descending.
        """
        if not documents:
            return []

        query_tokens = _tokenize(query)
        cfg = self._config
        hits: list[RecallHit] = []

        for doc in documents:
            # Signal 1: Keyword match
            doc_tokens = _tokenize(doc.text)
            kw_score = _keyword_score(query_tokens, doc_tokens)

            # Signal 2: Semantic similarity
            sem_score = 0.0
            if query_vector and doc.vector:
                sem_score = _cosine_similarity(query_vector, doc.vector)

            # Signal 3: Recency
            recency = _recency_factor(
                doc.age_days, cfg.recency_half_life_days
            )

            # Combine signals
            # If no query vector, redistribute semantic weight to keyword
            if query_vector and doc.vector:
                combined = (
                    cfg.keyword_weight * kw_score
                    + cfg.semantic_weight * sem_score
                )
            else:
                combined = kw_score  # keyword-only mode

            final_score = combined * recency

            if final_score >= cfg.min_score:
                hits.append(
                    RecallHit(
                        document_id=doc.id,
                        score=round(final_score, 6),
                        keyword_score=round(kw_score, 6),
                        semantic_score=round(sem_score, 6),
                        recency_factor=round(recency, 6),
                        metadata=doc.metadata,
                    )
                )

        # Sort by score descending
        hits.sort(key=lambda h: -h.score)

        return hits[: cfg.max_results]

    def search_keyword_only(
        self,
        query: str,
        documents: list[RecallDocument],
    ) -> list[RecallHit]:
        """Search using keywords and recency only (no vectors)."""
        return self.search(query=query, documents=documents, query_vector=())

    def search_semantic_only(
        self,
        query_vector: tuple[float, ...],
        documents: list[RecallDocument],
    ) -> list[RecallHit]:
        """Search using semantic similarity and recency only."""
        if not documents:
            return []

        cfg = self._config
        hits: list[RecallHit] = []

        for doc in documents:
            if not doc.vector:
                continue

            sem_score = _cosine_similarity(query_vector, doc.vector)
            recency = _recency_factor(
                doc.age_days, cfg.recency_half_life_days
            )
            final_score = sem_score * recency

            if final_score >= cfg.min_score:
                hits.append(
                    RecallHit(
                        document_id=doc.id,
                        score=round(final_score, 6),
                        keyword_score=0.0,
                        semantic_score=round(sem_score, 6),
                        recency_factor=round(recency, 6),
                        metadata=doc.metadata,
                    )
                )

        hits.sort(key=lambda h: -h.score)
        return hits[: cfg.max_results]
