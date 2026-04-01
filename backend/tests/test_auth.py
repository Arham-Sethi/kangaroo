"""Authentication endpoint tests.

Tests the complete auth flow:
    - Registration (happy path + duplicate email + validation)
    - Login (happy path + wrong password + email enumeration prevention)
    - Token refresh (happy path + invalid token types)
    - Protected endpoint access (valid token + expired + missing)

Every test is independent — the conftest provides a fresh database
session with automatic rollback after each test.
"""

import pytest


# -- Registration Tests -------------------------------------------------------


@pytest.mark.asyncio
async def test_register_success(client):
    """New user registration returns JWT tokens and 201 status."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "SecurePass123!",
            "display_name": "Test User",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    """Duplicate email returns 409 Conflict."""
    payload = {"email": "dupe@example.com", "password": "SecurePass123!"}
    first = await client.post("/api/v1/auth/register", json=payload)
    assert first.status_code == 201

    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_weak_password(client):
    """Password shorter than 8 chars is rejected with 422."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "short"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_invalid_email(client):
    """Invalid email format is rejected with 422."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "SecurePass123!"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_email_normalized(client):
    """Email is normalized to lowercase during registration."""
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": "UPPER@EXAMPLE.COM", "password": "SecurePass123!"},
    )
    assert response.status_code == 201

    token = response.json()["access_token"]
    profile = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert profile.json()["email"] == "upper@example.com"


# -- Login Tests --------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success(client):
    """Valid credentials return JWT tokens."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "login@example.com", "password": "SecurePass123!"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "SecurePass123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    """Wrong password returns 401 with generic error message."""
    await client.post(
        "/api/v1/auth/register",
        json={"email": "wrongpw@example.com", "password": "SecurePass123!"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpw@example.com", "password": "WrongPassword!"},
    )
    assert response.status_code == 401
    # Must NOT reveal whether email exists (enumeration prevention)
    assert response.json()["detail"] == "Invalid email or password."


@pytest.mark.asyncio
async def test_login_nonexistent_email(client):
    """Non-existent email returns same 401 as wrong password.

    This prevents email enumeration attacks — an attacker cannot
    determine whether an email is registered by observing different
    error messages or response times.
    """
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "ghost@example.com", "password": "AnyPassword123!"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


# -- Token Refresh Tests ------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_token_success(client):
    """Valid refresh token returns new token pair."""
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": "refresh@example.com", "password": "SecurePass123!"},
    )
    refresh_token = reg.json()["refresh_token"]

    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_refresh_with_access_token_rejected(client):
    """Using an access token for refresh is rejected."""
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": "badrefresh@example.com", "password": "SecurePass123!"},
    )
    access_token = reg.json()["access_token"]

    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": access_token},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_with_garbage_token(client):
    """Garbage token returns 401."""
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "not.a.real.token"},
    )
    assert response.status_code == 401


# -- Protected Endpoint Tests -------------------------------------------------


@pytest.mark.asyncio
async def test_protected_endpoint_with_token(client):
    """Access /auth/me with valid token returns user profile."""
    reg = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "protected@example.com",
            "password": "SecurePass123!",
            "display_name": "Protected User",
        },
    )
    token = reg.json()["access_token"]

    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "protected@example.com"
    assert data["display_name"] == "Protected User"
    assert data["subscription_tier"] == "free"
    assert data["shifts_this_month"] == 0


@pytest.mark.asyncio
async def test_protected_endpoint_no_token(client):
    """/auth/me without token returns 403."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_protected_endpoint_invalid_token(client):
    """/auth/me with garbage token returns 401."""
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer garbage.token.here"},
    )
    assert response.status_code == 401


# -- Middleware Tests ---------------------------------------------------------


@pytest.mark.asyncio
async def test_health_no_auth_required(client):
    """Health endpoint remains accessible without auth."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_request_id_in_response(client):
    """Every response includes X-Request-ID for tracing."""
    response = await client.get("/health")
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_process_time_in_response(client):
    """Every response includes X-Process-Time for latency monitoring."""
    response = await client.get("/health")
    assert "X-Process-Time" in response.headers
    # Must be a valid number
    float(response.headers["X-Process-Time"])


@pytest.mark.asyncio
async def test_custom_request_id_forwarded(client):
    """Client-provided X-Request-ID is echoed back."""
    custom_id = "test-request-12345"
    response = await client.get(
        "/health",
        headers={"X-Request-ID": custom_id},
    )
    assert response.headers["X-Request-ID"] == custom_id
