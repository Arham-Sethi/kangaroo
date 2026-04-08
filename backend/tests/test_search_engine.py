"""Tests for SearchEngine — in-memory BM25 keyword search.

Tests cover:
    - Document indexing
    - BM25 keyword search
    - Metadata filtering
    - Pagination
    - Document removal
    - Snippet extraction
    - Empty/edge cases
    - Clear operation
"""

import pytest

from app.core.storage.search import (
    SearchDocument,
    SearchEngine,
    SearchResponse,
    SearchResult,
    _tokenize,
)


# -- Tokenizer tests --------------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_empty(self) -> None:
        assert _tokenize("") == []


# -- Index tests -------------------------------------------------------------


class TestSearchEngineIndex:
    def test_index_documents(self) -> None:
        engine = SearchEngine()
        docs = [
            SearchDocument(id="d1", content="Python web development"),
            SearchDocument(id="d2", content="Database design patterns"),
        ]
        count = engine.index(docs)
        assert count == 2
        assert engine.document_count == 2

    def test_index_one(self) -> None:
        engine = SearchEngine()
        engine.index_one(SearchDocument(id="d1", content="test doc"))
        assert engine.document_count == 1

    def test_index_replaces_existing(self) -> None:
        engine = SearchEngine()
        engine.index([SearchDocument(id="d1", content="original")])
        engine.index([SearchDocument(id="d1", content="updated")])
        assert engine.document_count == 1

    def test_remove_document(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="test"),
            SearchDocument(id="d2", content="test"),
        ])
        assert engine.remove("d1") is True
        assert engine.document_count == 1

    def test_remove_nonexistent(self) -> None:
        engine = SearchEngine()
        assert engine.remove("nope") is False

    def test_clear(self) -> None:
        engine = SearchEngine()
        engine.index([SearchDocument(id="d1", content="test")])
        engine.clear()
        assert engine.document_count == 0


# -- Search tests ------------------------------------------------------------


class TestSearchEngineSearch:
    def test_basic_search(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="Python web development framework"),
            SearchDocument(id="d2", content="Database design patterns SQL"),
            SearchDocument(id="d3", content="Python REST API architecture"),
        ])
        result = engine.search("Python API")
        assert isinstance(result, SearchResponse)
        assert result.total >= 1
        assert result.query == "Python API"

    def test_empty_query(self) -> None:
        engine = SearchEngine()
        engine.index([SearchDocument(id="d1", content="test")])
        result = engine.search("")
        assert result.total == 0
        assert result.hits == []

    def test_no_matches(self) -> None:
        engine = SearchEngine()
        engine.index([SearchDocument(id="d1", content="Python web")])
        result = engine.search("quantum physics")
        assert result.total == 0

    def test_results_sorted_by_score(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="Python Python Python web API API"),
            SearchDocument(id="d2", content="single mention of Python"),
            SearchDocument(id="d3", content="Database SQL queries no python"),
        ])
        result = engine.search("Python API")
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_hit_has_snippet(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="Python is a great programming language for web APIs"),
        ])
        result = engine.search("Python")
        assert len(result.hits) > 0
        assert "Python" in result.hits[0].snippet or "python" in result.hits[0].snippet.lower()

    def test_hit_has_metadata(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="test", metadata={"key": "val"}),
        ])
        result = engine.search("test")
        assert len(result.hits) > 0
        assert result.hits[0].metadata == {"key": "val"}


# -- Filter tests ------------------------------------------------------------


class TestSearchEngineFilters:
    def test_metadata_filter(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="Python API", metadata={"user_id": "a"}),
            SearchDocument(id="d2", content="Python API", metadata={"user_id": "b"}),
        ])
        result = engine.search("Python", filters={"user_id": "a"})
        assert result.total == 1
        assert result.hits[0].id == "d1"

    def test_filter_no_matches(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id="d1", content="Python", metadata={"user_id": "a"}),
        ])
        result = engine.search("Python", filters={"user_id": "nonexistent"})
        assert result.total == 0


# -- Pagination tests --------------------------------------------------------


class TestSearchEnginePagination:
    def test_page_size(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id=f"d{i}", content="Python web API")
            for i in range(10)
        ])
        result = engine.search("Python", page=1, page_size=3)
        assert len(result.hits) <= 3
        assert result.page == 1
        assert result.page_size == 3
        assert result.total == 10

    def test_second_page(self) -> None:
        engine = SearchEngine()
        engine.index([
            SearchDocument(id=f"d{i}", content="Python web API")
            for i in range(10)
        ])
        p1 = engine.search("Python", page=1, page_size=3)
        p2 = engine.search("Python", page=2, page_size=3)

        p1_ids = {h.id for h in p1.hits}
        p2_ids = {h.id for h in p2.hits}
        # Pages should not overlap
        assert p1_ids.isdisjoint(p2_ids)

    def test_beyond_last_page(self) -> None:
        engine = SearchEngine()
        engine.index([SearchDocument(id="d1", content="Python")])
        result = engine.search("Python", page=100, page_size=10)
        assert result.hits == []
        assert result.total == 1
