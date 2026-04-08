"""Tests for EmbeddingEngine — TF-IDF embedding generation and vector search.

Tests cover:
    - Single text embedding
    - Vocabulary building from corpus
    - L2 normalization of vectors
    - Batch embedding
    - Cosine similarity search
    - Auto-vocabulary from single text
    - Empty input handling
    - Search result ordering and top_k
"""

import math

import pytest

from app.core.storage.embeddings import (
    EmbeddingEngine,
    EmbeddingResult,
    SearchHit,
    _cosine_similarity,
    _tokenize,
)


# -- Tokenizer tests --------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_strips_punctuation(self) -> None:
        tokens = _tokenize("Python, FastAPI, and REST!")
        assert "python" in tokens
        assert "fastapi" in tokens
        assert "rest" in tokens
        assert "," not in tokens

    def test_empty(self) -> None:
        assert _tokenize("") == []

    def test_all_punctuation(self) -> None:
        assert _tokenize("!@#$%^&*()") == []


# -- Cosine similarity tests ------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = (1.0, 0.0, 1.0)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self) -> None:
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self) -> None:
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_different_lengths(self) -> None:
        a = (1.0, 2.0)
        b = (1.0, 2.0, 3.0)
        assert _cosine_similarity(a, b) == 0.0

    def test_zero_vector(self) -> None:
        a = (0.0, 0.0)
        b = (1.0, 2.0)
        assert _cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity((), ()) == 0.0


# -- EmbeddingEngine tests --------------------------------------------------


class TestEmbeddingEngine:
    def test_embed_returns_result(self) -> None:
        engine = EmbeddingEngine()
        result = engine.embed("Python is great")
        assert isinstance(result, EmbeddingResult)
        assert result.text == "Python is great"
        assert len(result.vector) == result.dimension
        assert result.dimension > 0

    def test_embed_vector_is_normalized(self) -> None:
        engine = EmbeddingEngine()
        result = engine.embed("Python FastAPI REST API backend")
        norm = math.sqrt(sum(v * v for v in result.vector))
        if norm > 0:
            assert norm == pytest.approx(1.0, abs=1e-6)

    def test_auto_vocabulary_from_single_text(self) -> None:
        engine = EmbeddingEngine()
        assert engine.dimension == 0
        engine.embed("hello world")
        assert engine.dimension > 0

    def test_build_vocabulary(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary([
            "Python is a programming language",
            "FastAPI is a web framework",
            "REST APIs use HTTP methods",
        ])
        assert engine.dimension > 0
        assert engine.dimension <= 512

    def test_build_vocabulary_max_terms(self) -> None:
        engine = EmbeddingEngine()
        texts = [f"word{i} appears here" for i in range(100)]
        engine.build_vocabulary(texts, max_terms=10)
        assert engine.dimension <= 10

    def test_embed_with_vocabulary(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary([
            "Python programming language",
            "REST API design",
        ])
        result = engine.embed("Python API")
        assert result.dimension == engine.dimension

    def test_embed_batch(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary(["hello world", "foo bar"])
        results = engine.embed_batch(["hello", "world", "foo"])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EmbeddingResult)

    def test_similar_texts_have_high_similarity(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary([
            "Python web framework for REST APIs",
            "Python backend API development",
            "Cooking recipes for dinner",
        ])
        a = engine.embed("Python REST API framework")
        b = engine.embed("Python backend API development")
        c = engine.embed("Cooking recipes for dinner")

        sim_ab = _cosine_similarity(a.vector, b.vector)
        sim_ac = _cosine_similarity(a.vector, c.vector)

        # Related texts should be more similar than unrelated
        assert sim_ab > sim_ac

    def test_fixed_vocabulary(self) -> None:
        engine = EmbeddingEngine(vocabulary=["python", "api", "rest"])
        assert engine.dimension == 3
        result = engine.embed("python rest api")
        assert result.dimension == 3


# -- Search tests -----------------------------------------------------------


class TestEmbeddingSearch:
    def test_search_returns_hits(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary([
            "Python web development",
            "Database design patterns",
            "REST API architecture",
        ])

        query = engine.embed("Python web API")
        docs = [
            ("doc1", engine.embed("Python web development").vector, {"title": "Python"}),
            ("doc2", engine.embed("Database design patterns").vector, {"title": "DB"}),
            ("doc3", engine.embed("REST API architecture").vector, {"title": "API"}),
        ]

        results = engine.search(query.vector, docs, top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchHit) for r in results)

    def test_search_sorted_by_score(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary([
            "Python web development",
            "Database design",
            "REST API",
        ])

        query = engine.embed("Python web")
        docs = [
            ("doc1", engine.embed("Python web development").vector, {}),
            ("doc2", engine.embed("Database design").vector, {}),
            ("doc3", engine.embed("REST API").vector, {}),
        ]

        results = engine.search(query.vector, docs)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary(["word"] * 5)

        query = engine.embed("word")
        docs = [
            (f"doc{i}", engine.embed("word").vector, {}) for i in range(10)
        ]

        results = engine.search(query.vector, docs, top_k=3)
        assert len(results) <= 3

    def test_search_empty_documents(self) -> None:
        engine = EmbeddingEngine()
        result = engine.search((1.0, 0.0), [], top_k=5)
        assert result == []

    def test_search_score_is_rounded(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary(["hello world"])
        query = engine.embed("hello")
        docs = [("d1", engine.embed("hello world").vector, {})]
        results = engine.search(query.vector, docs)
        for r in results:
            # Score should be rounded to 6 decimal places
            assert r.score == round(r.score, 6)

    def test_search_hit_has_metadata(self) -> None:
        engine = EmbeddingEngine()
        engine.build_vocabulary(["test data"])
        query = engine.embed("test")
        docs = [("d1", engine.embed("test data").vector, {"key": "value"})]
        results = engine.search(query.vector, docs)
        assert len(results) > 0
        assert results[0].metadata == {"key": "value"}
        assert results[0].id == "d1"
