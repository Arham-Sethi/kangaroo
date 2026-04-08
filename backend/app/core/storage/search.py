"""Full-text keyword search engine for session metadata.

Production: Uses Meilisearch for high-performance full-text search.
Fallback: Uses an in-memory inverted index for offline/test environments.

Indexes session metadata (titles, tags, entity names, decision descriptions)
and provides filtered, paginated keyword search.

Usage:
    engine = SearchEngine()
    engine.index(documents)
    results = engine.search("authentication decisions", filters={"user_id": "abc"})
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchDocument:
    """A document to be indexed for keyword search.

    Attributes:
        id: Unique document identifier.
        content: Searchable text content.
        metadata: Filterable metadata (user_id, session_id, etc.).
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A keyword search result.

    Attributes:
        id: Matched document ID.
        score: Relevance score (higher is better).
        snippet: Matching text snippet.
        metadata: Document metadata.
    """

    id: str
    score: float
    snippet: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResponse:
    """Paginated search response.

    Attributes:
        hits: List of matching documents.
        total: Total number of matches (before pagination).
        query: The original query string.
        page: Current page number (1-indexed).
        page_size: Number of results per page.
    """

    hits: list[SearchResult]
    total: int
    query: str
    page: int
    page_size: int


def _tokenize(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _unique_tokens(text: str) -> set[str]:
    """Extract unique lowercase tokens from text."""
    return set(_tokenize(text))


class SearchEngine:
    """In-memory keyword search engine with TF-IDF scoring.

    This is the fallback implementation. In production, calls are
    proxied to Meilisearch via the MeilisearchSearchEngine subclass.

    Features:
        - TF-IDF scoring with BM25 variant
        - Metadata filtering (exact match on any metadata field)
        - Pagination support
        - Snippet extraction (first 200 chars of matching content)
    """

    def __init__(self) -> None:
        self._documents: dict[str, SearchDocument] = {}
        self._inverted_index: dict[str, set[str]] = {}  # token -> doc_ids
        self._doc_token_counts: dict[str, dict[str, int]] = {}  # doc_id -> {token: count}
        self._doc_lengths: dict[str, int] = {}  # doc_id -> total tokens
        self._avg_doc_length: float = 0.0

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def index(self, documents: list[SearchDocument]) -> int:
        """Index a batch of documents.

        Args:
            documents: Documents to add to the index.

        Returns:
            Number of documents indexed.
        """
        indexed = 0
        for doc in documents:
            self._documents[doc.id] = doc

            tokens = _tokenize(doc.content)
            self._doc_lengths[doc.id] = len(tokens)

            # Count term frequencies
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_token_counts[doc.id] = tf

            # Update inverted index
            for token in set(tokens):
                if token not in self._inverted_index:
                    self._inverted_index[token] = set()
                self._inverted_index[token].add(doc.id)

            indexed += 1

        # Recompute average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(
                self._doc_lengths
            )

        return indexed

    def index_one(self, document: SearchDocument) -> None:
        """Index a single document."""
        self.index([document])

    def remove(self, document_id: str) -> bool:
        """Remove a document from the index.

        Returns True if the document was found and removed.
        """
        if document_id not in self._documents:
            return False

        del self._documents[document_id]

        # Clean inverted index
        tokens = self._doc_token_counts.pop(document_id, {})
        for token in tokens:
            if token in self._inverted_index:
                self._inverted_index[token].discard(document_id)
                if not self._inverted_index[token]:
                    del self._inverted_index[token]

        self._doc_lengths.pop(document_id, None)

        # Recompute average
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(
                self._doc_lengths
            )
        else:
            self._avg_doc_length = 0.0

        return True

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> SearchResponse:
        """Search indexed documents by keyword query.

        Args:
            query: Search query string.
            filters: Optional metadata filters (exact match).
            page: Page number (1-indexed).
            page_size: Results per page.

        Returns:
            Paginated SearchResponse.
        """
        if not query.strip():
            return SearchResponse(
                hits=[], total=0, query=query, page=page, page_size=page_size
            )

        query_tokens = _tokenize(query)
        if not query_tokens:
            return SearchResponse(
                hits=[], total=0, query=query, page=page, page_size=page_size
            )

        # Find candidate documents (union of token postings)
        candidate_ids: set[str] = set()
        for token in query_tokens:
            if token in self._inverted_index:
                candidate_ids |= self._inverted_index[token]

        if not candidate_ids:
            return SearchResponse(
                hits=[], total=0, query=query, page=page, page_size=page_size
            )

        # Apply metadata filters
        if filters:
            candidate_ids = {
                doc_id
                for doc_id in candidate_ids
                if self._matches_filters(doc_id, filters)
            }

        # Score candidates using BM25
        scored: list[tuple[float, str]] = []
        n = len(self._documents)
        k1 = 1.5
        b = 0.75

        for doc_id in candidate_ids:
            score = 0.0
            doc_tf = self._doc_token_counts.get(doc_id, {})
            doc_len = self._doc_lengths.get(doc_id, 0)

            for token in query_tokens:
                tf = doc_tf.get(token, 0)
                if tf == 0:
                    continue

                df = len(self._inverted_index.get(token, set()))
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

                if self._avg_doc_length > 0:
                    tf_norm = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * doc_len / self._avg_doc_length)
                    )
                else:
                    tf_norm = tf

                score += idf * tf_norm

            if score > 0:
                scored.append((score, doc_id))

        # Sort by score descending
        scored.sort(key=lambda x: -x[0])

        total = len(scored)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_results = scored[start:end]

        hits = [
            SearchResult(
                id=doc_id,
                score=round(score, 6),
                snippet=self._make_snippet(doc_id, query_tokens),
                metadata=self._documents[doc_id].metadata,
            )
            for score, doc_id in page_results
        ]

        return SearchResponse(
            hits=hits,
            total=total,
            query=query,
            page=page,
            page_size=page_size,
        )

    def _matches_filters(
        self, doc_id: str, filters: dict[str, Any]
    ) -> bool:
        """Check if a document's metadata matches all filters."""
        doc = self._documents.get(doc_id)
        if not doc:
            return False

        for key, value in filters.items():
            if doc.metadata.get(key) != value:
                return False

        return True

    def _make_snippet(self, doc_id: str, query_tokens: list[str]) -> str:
        """Extract a text snippet around the first matching token."""
        doc = self._documents.get(doc_id)
        if not doc:
            return ""

        content = doc.content
        content_lower = content.lower()

        # Find earliest match position
        best_pos = len(content)
        for token in query_tokens:
            pos = content_lower.find(token)
            if pos != -1 and pos < best_pos:
                best_pos = pos

        # Extract snippet around match
        start = max(0, best_pos - 50)
        end = min(len(content), best_pos + 150)

        snippet = content[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._documents.clear()
        self._inverted_index.clear()
        self._doc_token_counts.clear()
        self._doc_lengths.clear()
        self._avg_doc_length = 0.0
