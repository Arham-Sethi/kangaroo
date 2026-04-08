"""Tests for MemoryRecall — hybrid search combining keyword, semantic, recency.

Tests cover:
    - Keyword-only search
    - Semantic-only search
    - Hybrid search (keyword + semantic)
    - Recency decay effect
    - Min score filtering
    - Max results limiting
    - Empty input handling
    - Config customization
    - RecallHit score components
"""

import math

import pytest

from app.core.brain.recall import (
    MemoryRecall,
    RecallConfig,
    RecallDocument,
    RecallHit,
    _cosine_similarity,
    _keyword_score,
    _recency_factor,
    _tokenize,
)


# -- Helper functions --------------------------------------------------------


def _make_doc(
    doc_id: str = "d1",
    text: str = "test document",
    vector: tuple[float, ...] = (),
    age_days: float = 0.0,
    metadata: dict | None = None,
) -> RecallDocument:
    return RecallDocument(
        id=doc_id,
        text=text,
        vector=vector,
        age_days=age_days,
        metadata=metadata or {},
    )


# -- Tokenizer tests --------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_returns_set(self) -> None:
        tokens = _tokenize("word word word")
        assert isinstance(tokens, set)
        assert len(tokens) == 1

    def test_empty(self) -> None:
        assert _tokenize("") == set()


# -- Keyword score tests ----------------------------------------------------


class TestKeywordScore:
    def test_identical(self) -> None:
        tokens = {"python", "api"}
        assert _keyword_score(tokens, tokens) == 1.0

    def test_no_overlap(self) -> None:
        assert _keyword_score({"python"}, {"rust"}) == 0.0

    def test_partial_overlap(self) -> None:
        a = {"python", "api", "web"}
        b = {"python", "api", "database"}
        score = _keyword_score(a, b)
        # Jaccard: 2 / 4 = 0.5
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_empty_query(self) -> None:
        assert _keyword_score(set(), {"a"}) == 0.0

    def test_empty_doc(self) -> None:
        assert _keyword_score({"a"}, set()) == 0.0


# -- Cosine similarity tests ------------------------------------------------


class TestCosineSimilarity:
    def test_identical(self) -> None:
        v = (1.0, 0.0, 1.0)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_empty(self) -> None:
        assert _cosine_similarity((), ()) == 0.0

    def test_length_mismatch(self) -> None:
        assert _cosine_similarity((1.0,), (1.0, 2.0)) == 0.0


# -- Recency factor tests ---------------------------------------------------


class TestRecencyFactor:
    def test_age_zero(self) -> None:
        assert _recency_factor(0.0, 30.0) == 1.0

    def test_half_life(self) -> None:
        factor = _recency_factor(30.0, 30.0)
        assert factor == pytest.approx(0.5, abs=1e-6)

    def test_double_half_life(self) -> None:
        factor = _recency_factor(60.0, 30.0)
        assert factor == pytest.approx(0.25, abs=1e-6)

    def test_negative_age(self) -> None:
        assert _recency_factor(-1.0, 30.0) == 1.0

    def test_zero_half_life(self) -> None:
        assert _recency_factor(10.0, 0.0) == 1.0


# -- MemoryRecall keyword-only tests ----------------------------------------


class TestRecallKeywordOnly:
    def test_basic_keyword_search(self) -> None:
        recall = MemoryRecall()
        docs = [
            _make_doc("d1", "python web api development"),
            _make_doc("d2", "database design patterns"),
            _make_doc("d3", "python rest api framework"),
        ]
        hits = recall.search_keyword_only("python api", docs)
        assert len(hits) > 0
        # Python API docs should score higher
        doc_ids = [h.document_id for h in hits]
        assert "d1" in doc_ids or "d3" in doc_ids

    def test_no_matches(self) -> None:
        recall = MemoryRecall()
        docs = [_make_doc("d1", "completely unrelated content xyz")]
        hits = recall.search_keyword_only("quantum physics", docs)
        # May return empty or very low scores
        for h in hits:
            assert h.keyword_score < 0.5

    def test_empty_documents(self) -> None:
        recall = MemoryRecall()
        hits = recall.search_keyword_only("test", [])
        assert hits == []


# -- MemoryRecall semantic-only tests ----------------------------------------


class TestRecallSemanticOnly:
    def test_basic_semantic_search(self) -> None:
        recall = MemoryRecall()
        docs = [
            _make_doc("d1", vector=(1.0, 0.0, 0.0)),
            _make_doc("d2", vector=(0.0, 1.0, 0.0)),
            _make_doc("d3", vector=(0.9, 0.1, 0.0)),
        ]
        query_vec = (1.0, 0.0, 0.0)
        hits = recall.search_semantic_only(query_vec, docs)
        assert len(hits) > 0
        # d1 and d3 should score highest (closest to query)
        assert hits[0].document_id in ("d1", "d3")

    def test_skips_docs_without_vectors(self) -> None:
        recall = MemoryRecall()
        docs = [
            _make_doc("d1", vector=()),  # no vector
            _make_doc("d2", vector=(1.0, 0.0)),
        ]
        hits = recall.search_semantic_only((1.0, 0.0), docs)
        doc_ids = [h.document_id for h in hits]
        assert "d1" not in doc_ids

    def test_empty_documents(self) -> None:
        recall = MemoryRecall()
        hits = recall.search_semantic_only((1.0, 0.0), [])
        assert hits == []


# -- MemoryRecall hybrid tests ----------------------------------------------


class TestRecallHybrid:
    def test_hybrid_combines_signals(self) -> None:
        recall = MemoryRecall(config=RecallConfig(
            keyword_weight=0.5,
            semantic_weight=0.5,
        ))
        docs = [
            _make_doc("d1", text="python api", vector=(1.0, 0.0)),
            _make_doc("d2", text="database design", vector=(0.0, 1.0)),
        ]
        hits = recall.search(
            query="python api",
            documents=docs,
            query_vector=(1.0, 0.0),
        )
        assert len(hits) > 0
        # d1 matches both keyword AND semantic
        assert hits[0].document_id == "d1"
        assert hits[0].keyword_score > 0
        assert hits[0].semantic_score > 0

    def test_recency_boosts_newer(self) -> None:
        recall = MemoryRecall(config=RecallConfig(
            recency_half_life_days=30,
            keyword_weight=1.0,
            semantic_weight=0.0,
        ))
        docs = [
            _make_doc("old", text="python api", age_days=90),
            _make_doc("new", text="python api", age_days=0),
        ]
        hits = recall.search_keyword_only("python api", docs)
        assert len(hits) == 2
        # Both have same keyword score but new should rank higher
        assert hits[0].document_id == "new"
        assert hits[0].recency_factor > hits[1].recency_factor

    def test_min_score_filtering(self) -> None:
        recall = MemoryRecall(config=RecallConfig(min_score=0.5))
        docs = [
            _make_doc("d1", text="totally unrelated xyz abc"),
        ]
        hits = recall.search_keyword_only("python api", docs)
        # Low overlap should be filtered out
        assert len(hits) == 0

    def test_max_results(self) -> None:
        recall = MemoryRecall(config=RecallConfig(max_results=2))
        docs = [
            _make_doc(f"d{i}", text="python api web") for i in range(10)
        ]
        hits = recall.search_keyword_only("python api web", docs)
        assert len(hits) <= 2

    def test_scores_sorted_descending(self) -> None:
        recall = MemoryRecall()
        docs = [
            _make_doc("d1", text="python", vector=(1.0, 0.0)),
            _make_doc("d2", text="python api web", vector=(0.8, 0.2)),
            _make_doc("d3", text="python api web framework rest", vector=(0.9, 0.1)),
        ]
        hits = recall.search(
            query="python api web",
            documents=docs,
            query_vector=(0.9, 0.1),
        )
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_hit_has_all_components(self) -> None:
        recall = MemoryRecall()
        docs = [_make_doc("d1", text="python api", vector=(1.0, 0.0))]
        hits = recall.search("python", docs, query_vector=(1.0, 0.0))
        assert len(hits) > 0
        hit = hits[0]
        assert isinstance(hit, RecallHit)
        assert hit.document_id == "d1"
        assert hit.score >= 0
        assert hit.keyword_score >= 0
        assert hit.semantic_score >= 0
        assert hit.recency_factor >= 0

    def test_metadata_preserved(self) -> None:
        recall = MemoryRecall()
        docs = [_make_doc("d1", text="python", metadata={"key": "val"})]
        hits = recall.search_keyword_only("python", docs)
        assert len(hits) > 0
        assert hits[0].metadata == {"key": "val"}

    def test_without_query_vector_uses_keyword_only(self) -> None:
        recall = MemoryRecall()
        docs = [_make_doc("d1", text="python api", vector=(1.0, 0.0))]
        hits = recall.search("python", docs, query_vector=())
        for h in hits:
            assert h.semantic_score == 0.0


# -- Config tests -----------------------------------------------------------


class TestRecallConfig:
    def test_default_config(self) -> None:
        cfg = RecallConfig()
        assert cfg.keyword_weight == 0.4
        assert cfg.semantic_weight == 0.6
        assert cfg.recency_half_life_days == 30.0
        assert cfg.min_score == 0.01
        assert cfg.max_results == 20

    def test_custom_config(self) -> None:
        cfg = RecallConfig(
            keyword_weight=0.7,
            semantic_weight=0.3,
            recency_half_life_days=7.0,
            min_score=0.1,
            max_results=5,
        )
        recall = MemoryRecall(config=cfg)
        assert recall.config.keyword_weight == 0.7
        assert recall.config.max_results == 5
