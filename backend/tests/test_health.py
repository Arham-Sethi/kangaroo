"""Health endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint returns ok status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
