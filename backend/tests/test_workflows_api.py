"""Tests for the workflow orchestration API endpoints.

Tests cover:
    - Workflow CRUD: create, list, get, delete
    - Chain execution: sequential model pipeline
    - Consensus execution: parallel model comparison
    - Input validation: missing fields, wrong types
    - Authorization: ownership checks
    - Run history: listing past executions
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.v1.workflows import (
    router,
    _workflows,
    _workflow_runs,
)
from app.core.cockpit.orchestrator import ModelResponse


# ── Test fixtures ──────────────────────────────────────────────────────────


def _build_test_app() -> FastAPI:
    """Create a minimal FastAPI app with workflows router."""
    app = FastAPI()
    app.include_router(router, prefix="/workflows")
    return app


def _mock_user(sub: str = "user-1"):
    """Return a mock get_current_user dependency override."""
    async def _override():
        return {"sub": sub}
    return _override


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clear in-memory stores before each test."""
    _workflows.clear()
    _workflow_runs.clear()
    yield
    _workflows.clear()
    _workflow_runs.clear()


@pytest.fixture
def app():
    """FastAPI test app with auth override."""
    from app.api.v1.auth import get_current_user

    test_app = _build_test_app()
    test_app.dependency_overrides[get_current_user] = _mock_user("user-1")
    return test_app


@pytest.fixture
async def client(app: FastAPI):
    """Async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Chain Workflow CRUD ────────────────────────────────────────────────────


class TestCreateWorkflow:
    @pytest.mark.asyncio
    async def test_create_chain_workflow(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "Draft & Review",
            "workflow_type": "chain",
            "description": "Two-step pipeline",
            "steps": [
                {"model_id": "gpt-4o", "instruction": "Write a draft"},
                {"model_id": "claude-sonnet-4", "instruction": "Review"},
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Draft & Review"
        assert data["workflow_type"] == "chain"
        assert len(data["steps"]) == 2
        assert data["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_create_consensus_workflow(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "Fact Check",
            "workflow_type": "consensus",
            "models": ["gpt-4o", "claude-sonnet-4", "gemini-2.0-flash"],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["workflow_type"] == "consensus"
        assert len(data["models"]) == 3

    @pytest.mark.asyncio
    async def test_create_chain_requires_steps(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "Empty Chain",
            "workflow_type": "chain",
            "steps": [],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_consensus_requires_2_models(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "Single Model",
            "workflow_type": "consensus",
            "models": ["gpt-4o"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_rejects_invalid_type(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "Bad Type",
            "workflow_type": "invalid",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_rejects_empty_name(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows", json={
            "name": "",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Draft"}],
        })
        assert resp.status_code == 422


class TestListWorkflows:
    @pytest.mark.asyncio
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/workflows")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_returns_user_workflows(self, client: AsyncClient) -> None:
        # Create two workflows
        await client.post("/workflows", json={
            "name": "First",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Go"}],
        })
        await client.post("/workflows", json={
            "name": "Second",
            "workflow_type": "consensus",
            "models": ["gpt-4o", "claude-sonnet-4"],
        })

        resp = await client.get("/workflows")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_filters_by_user(self, app: FastAPI, client: AsyncClient) -> None:
        from app.api.v1.auth import get_current_user

        # Create workflow as user-1
        await client.post("/workflows", json={
            "name": "User 1 Workflow",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Go"}],
        })

        # Switch to user-2
        app.dependency_overrides[get_current_user] = _mock_user("user-2")

        resp = await client.get("/workflows")
        assert resp.status_code == 200
        assert len(resp.json()) == 0


class TestGetWorkflow:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "Test",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Go"}],
        })
        wf_id = create_resp.json()["id"]

        resp = await client.get(f"/workflows/{wf_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == wf_id

    @pytest.mark.asyncio
    async def test_get_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/workflows/nonexistent")
        assert resp.status_code == 404


class TestDeleteWorkflow:
    @pytest.mark.asyncio
    async def test_delete_existing(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "To Delete",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Go"}],
        })
        wf_id = create_resp.json()["id"]

        resp = await client.delete(f"/workflows/{wf_id}")
        assert resp.status_code == 204

        # Verify gone
        resp = await client.get(f"/workflows/{wf_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, client: AsyncClient) -> None:
        resp = await client.delete("/workflows/nonexistent")
        assert resp.status_code == 404


# ── Workflow Execution ──────────────────────────────────────────────────────


class TestRunChainWorkflow:
    @pytest.mark.asyncio
    async def test_run_chain(self, client: AsyncClient) -> None:
        # Create a chain workflow
        create_resp = await client.post("/workflows", json={
            "name": "Chain",
            "workflow_type": "chain",
            "steps": [
                {"model_id": "gpt-4o", "instruction": "Draft"},
                {"model_id": "claude-sonnet-4", "instruction": "Review"},
            ],
        })
        wf_id = create_resp.json()["id"]

        # Mock the chain executor to avoid real LLM calls
        mock_response = ModelResponse(
            model_id="gpt-4o",
            content="Mocked chain output",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
        )

        with patch(
            "app.core.cockpit.chain._default_chain_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = await client.post(f"/workflows/{wf_id}/run", json={
                "prompt": "Write a hello world",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["workflow_type"] == "chain"
        assert "steps" in data["result"]

    @pytest.mark.asyncio
    async def test_run_not_found(self, client: AsyncClient) -> None:
        resp = await client.post("/workflows/nonexistent/run", json={
            "prompt": "Test",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_empty_prompt(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "Chain",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Draft"}],
        })
        wf_id = create_resp.json()["id"]

        resp = await client.post(f"/workflows/{wf_id}/run", json={
            "prompt": "",
        })
        assert resp.status_code == 422


class TestRunConsensusWorkflow:
    @pytest.mark.asyncio
    async def test_run_consensus(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "Consensus",
            "workflow_type": "consensus",
            "models": ["gpt-4o", "claude-sonnet-4"],
        })
        wf_id = create_resp.json()["id"]

        mock_response = ModelResponse(
            model_id="gpt-4o",
            content="Mocked consensus output with lots of words for comparison",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
        )

        with patch(
            "app.core.cockpit.consensus._default_consensus_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            resp = await client.post(f"/workflows/{wf_id}/run", json={
                "prompt": "What is the best database?",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "agreement_level" in data["result"]


class TestWorkflowRuns:
    @pytest.mark.asyncio
    async def test_list_runs_empty(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "Chain",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Draft"}],
        })
        wf_id = create_resp.json()["id"]

        resp = await client.get(f"/workflows/{wf_id}/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_runs_after_execution(self, client: AsyncClient) -> None:
        create_resp = await client.post("/workflows", json={
            "name": "Chain",
            "workflow_type": "chain",
            "steps": [{"model_id": "gpt-4o", "instruction": "Draft"}],
        })
        wf_id = create_resp.json()["id"]

        mock_response = ModelResponse(
            model_id="gpt-4o",
            content="Output",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
        )

        with patch(
            "app.core.cockpit.chain._default_chain_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            await client.post(f"/workflows/{wf_id}/run", json={"prompt": "Test"})

        resp = await client.get(f"/workflows/{wf_id}/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["workflow_id"] == wf_id

    @pytest.mark.asyncio
    async def test_list_runs_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/workflows/nonexistent/runs")
        assert resp.status_code == 404
