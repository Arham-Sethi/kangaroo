"""Embedding generation and vector similarity search.

Generates embeddings for UCS content (entities, decisions, summaries)
and provides vector similarity search across sessions.

Production: Uses sentence-transformers for high-quality embeddings.
Fallback: Uses TF-IDF bag-of-words for offline/test environments.

Usage:
    engine = EmbeddingEngine()
    vec = engine.embed("What did I decide about authentication?")
    results = engine.search(query_vec, stored_vectors, top_k=5)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of embedding a text.

    Attributes:
        text: The original text.
        vector: The embedding vector.
        dimension: Vector dimensionality.
    """

    text: str
    vector: tuple[float, ...]
    dimension: int


@dataclass(frozen=True)
class SearchHit:
    """A search result with similarity score.

    Attributes:
        id: Identifier of the matched document.
        score: Cosine similarity score (0..1).
        metadata: Additional metadata about the match.
    """

    id: str
    score: float
    metadata: dict[str, Any]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class EmbeddingEngine:
    """Generates text embeddings and performs vector similarity search.

    Uses a bag-of-words TF-IDF approach as the default (no external deps).
    In production, swap with sentence-transformers by overriding embed().
    """

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        """Initialize the embedding engine.

        Args:
            vocabulary: Fixed vocabulary for consistent vector dimensions.
                       If None, vocabulary is built from the first batch of texts.
        """
        self._vocab: list[str] = vocabulary or []
        self._vocab_index: dict[str, int] = {
            w: i for i, w in enumerate(self._vocab)
        }
        self._idf: dict[str, float] = {}

    @property
    def dimension(self) -> int:
        return len(self._vocab)

    def build_vocabulary(self, texts: list[str], max_terms: int = 512) -> None:
        """Build vocabulary from a corpus of texts.

        Args:
            texts: Corpus of documents.
            max_terms: Maximum vocabulary size.
        """
        doc_freq: dict[str, int] = {}
        total_docs = len(texts)

        for text in texts:
            tokens = set(_tokenize(text))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Sort by document frequency descending, take top N
        sorted_terms = sorted(doc_freq.items(), key=lambda x: -x[1])
        self._vocab = [term for term, _ in sorted_terms[:max_terms]]
        self._vocab_index = {w: i for i, w in enumerate(self._vocab)}

        # Compute IDF
        self._idf = {}
        for term, df in doc_freq.items():
            if term in self._vocab_index:
                self._idf[term] = math.log((total_docs + 1) / (df + 1)) + 1

    def embed(self, text: str) -> EmbeddingResult:
        """Generate an embedding vector for a text.

        Returns a TF-IDF vector over the vocabulary.
        """
        if not self._vocab:
            # Auto-build vocabulary from this single text
            tokens = list(set(_tokenize(text)))
            self._vocab = sorted(tokens)[:512]
            self._vocab_index = {w: i for i, w in enumerate(self._vocab)}

        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        vector = []
        for term in self._vocab:
            term_tf = tf.get(term, 0)
            term_idf = self._idf.get(term, 1.0)
            vector.append(term_tf * term_idf)

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return EmbeddingResult(
            text=text,
            vector=tuple(vector),
            dimension=len(vector),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts."""
        return [self.embed(text) for text in texts]

    def search(
        self,
        query_vector: tuple[float, ...],
        documents: list[tuple[str, tuple[float, ...], dict[str, Any]]],
        top_k: int = 10,
    ) -> list[SearchHit]:
        """Find the most similar documents to a query vector.

        Args:
            query_vector: The query embedding.
            documents: List of (id, vector, metadata) tuples.
            top_k: Number of results to return.

        Returns:
            List of SearchHit sorted by similarity descending.
        """
        scored: list[tuple[float, str, dict[str, Any]]] = []

        for doc_id, doc_vector, metadata in documents:
            score = _cosine_similarity(query_vector, doc_vector)
            scored.append((score, doc_id, metadata))

        scored.sort(key=lambda x: -x[0])

        return [
            SearchHit(id=doc_id, score=round(score, 6), metadata=meta)
            for score, doc_id, meta in scored[:top_k]
        ]
