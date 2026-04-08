"""Tests for search API endpoints — keyword, semantic, and recall search.

Tests cover:
    - GET /api/v1/search — keyword search
    - POST /api/v1/search/semantic — semantic search
    - POST /api/v1/search/recall — hybrid recall search
    - Authentication required
    - Empty results
    - Pagination
    - User isolation (can't search other user's sessions)
"""

import uuid

import pytest


# -- Helpers -----------------------------------------------------------------


def _unique_email() -> str:
    return f"test-{uuid.uuid4().hex[:12]}@example.com"


async def _register_and_get_token(client, email: str | None = None):
    if email is None:
        email = _unique_email()
    resp = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "SecurePass123!"},
    )
    assert resp.status_code == 201
    return resp.json()["access_token"]


async def _create_session(client, token, **overrides):
    body = {
        "title": "Test Session",
        "source_llm": "openai",
        "source_model": "gpt-4o",
        "message_count": 10,
        "total_tokens": 1500,
        "tags": ["test"],
    }
    body.update(overrides)
    resp = await client.post(
        "/api/v1/sessions",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201
    return resp.json()


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# -- Keyword Search Tests ---------------------------------------------------


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.get("/api/v1/search?q=test")
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get(
            "/api/v1/search?q=",
            headers=_auth(token),
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_sessions_returns_empty(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.get(
            "/api/v1/search?q=python",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["hits"] == []

    @pytest.mark.asyncio
    async def test_finds_matching_session(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(
            client, token,
            title="Python REST API Design",
            tags=["python", "api"],
        )
        await _create_session(
            client, token,
            title="Database Schema Migration",
            tags=["database"],
        )

        resp = await client.get(
            "/api/v1/search?q=python+api",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        assert any("Python" in h["metadata"].get("title", "") for h in data["hits"])

    @pytest.mark.asyncio
    async def test_pagination(self, client) -> None:
        token = await _register_and_get_token(client)
        for i in range(5):
            await _create_session(
                client, token,
                title=f"Python Tutorial Part {i}",
                tags=["python"],
            )

        resp = await client.get(
            "/api/v1/search?q=python&page=1&page_size=2",
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hits"]) <= 2
        assert data["page"] == 1
        assert data["page_size"] == 2

    @pytest.mark.asyncio
    async def test_user_isolation(self, client) -> None:
        token_a = await _register_and_get_token(client)
        token_b = await _register_and_get_token(client)

        await _create_session(client, token_a, title="Secret Python Project")
        await _create_session(client, token_b, title="Rust Database Design")

        # User B should not find User A's session
        resp = await client.get(
            "/api/v1/search?q=secret+python",
            headers=_auth(token_b),
        )
        assert resp.status_code == 200
        data = resp.json()
        for hit in data["hits"]:
            assert "Secret" not in hit["metadata"].get("title", "")


# -- Semantic Search Tests --------------------------------------------------


class TestSemanticSearch:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/search/semantic",
            json={"query": "test"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_sessions_returns_empty(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/search/semantic",
            json={"query": "python api", "top_k": 5},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_finds_similar_sessions(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(
            client, token,
            title="Python FastAPI Backend",
            tags=["python", "fastapi"],
        )
        await _create_session(
            client, token,
            title="React Frontend Design",
            tags=["react", "frontend"],
        )

        resp = await client.post(
            "/api/v1/search/semantic",
            json={"query": "python backend api", "top_k": 10},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_validation_empty_query(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/search/semantic",
            json={"query": "", "top_k": 5},
            headers=_auth(token),
        )
        assert resp.status_code == 422


# -- Recall Search Tests ----------------------------------------------------


class TestRecallSearch:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client) -> None:
        resp = await client.post(
            "/api/v1/search/recall",
            json={"query": "test"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_sessions_returns_empty(self, client) -> None:
        token = await _register_and_get_token(client)
        resp = await client.post(
            "/api/v1/search/recall",
            json={"query": "auth decisions"},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["hits"] == []

    @pytest.mark.asyncio
    async def test_hybrid_recall_returns_hits(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(
            client, token,
            title="Authentication Design Decisions",
            tags=["auth", "security"],
        )
        await _create_session(
            client, token,
            title="Database Performance Tuning",
            tags=["database", "performance"],
        )

        resp = await client.post(
            "/api/v1/search/recall",
            json={"query": "authentication decisions", "max_results": 10},
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        # Each hit should have score components
        for hit in data["hits"]:
            assert "score" in hit
            assert "keyword_score" in hit
            assert "semantic_score" in hit
            assert "recency_factor" in hit

    @pytest.mark.asyncio
    async def test_recall_custom_weights(self, client) -> None:
        token = await _register_and_get_token(client)
        await _create_session(
            client, token,
            title="Python API Development",
            tags=["python"],
        )

        resp = await client.post(
            "/api/v1/search/recall",
            json={
                "query": "python api",
                "keyword_weight": 0.8,
                "semantic_weight": 0.2,
                "max_results": 5,
            },
            headers=_auth(token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    @pytest.mark.asyncio
    async def test_recall_validation(self, client) -> None:
        token = await _register_and_get_token(client)
        # Invalid weight
        resp = await client.post(
            "/api/v1/search/recall",
            json={"query": "test", "keyword_weight": 2.0},
            headers=_auth(token),
        )
        assert resp.status_code == 422
