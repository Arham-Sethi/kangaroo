"""API request/response Pydantic schemas.

These schemas define the public API contract. They are separate from the
database models (db.py) and the UCS format (ucs.py) because:
    1. API responses should never expose internal fields (password_hash, etc.)
    2. Request schemas enforce input validation at the API boundary
    3. Response schemas control exactly what data leaves the system
    4. Decoupling allows API and database schemas to evolve independently

Naming convention:
    - {Resource}Create — POST request body
    - {Resource}Update — PATCH request body
    - {Resource}Response — Response body
    - {Resource}ListResponse — Paginated list response
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr


# ── Auth Schemas ──────────────────────────────────────────────────────────────


class UserRegister(BaseModel):
    """Registration request. Email + password is all we need to start."""

    email: EmailStr
    password: SecretStr = Field(min_length=8, max_length=128)
    display_name: str = Field(default="", max_length=100)


class UserLogin(BaseModel):
    """Login request."""

    email: EmailStr
    password: SecretStr


class TokenResponse(BaseModel):
    """JWT token pair returned after login/register/refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds.")


class TokenRefresh(BaseModel):
    """Refresh token request."""

    refresh_token: str


class UserResponse(BaseModel):
    """Public user profile. Never exposes password_hash or internal fields."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    display_name: str
    subscription_tier: str
    is_verified: bool
    shifts_this_month: int
    created_at: datetime
    settings: dict


# ── Session Schemas ───────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    """Create a new conversation session."""

    title: str = Field(default="Untitled Session", max_length=500)
    source_llm: str = Field(max_length=50)
    source_model: str = Field(default="", max_length=100)
    metadata: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    """Update session metadata (title, tags, archive status)."""

    title: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = None
    is_archived: bool | None = None
    metadata: dict | None = None


class SessionResponse(BaseModel):
    """Session data returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    source_llm: str
    source_model: str
    message_count: int
    total_tokens: int
    tags: list
    metadata: dict
    is_archived: bool
    created_at: datetime
    updated_at: datetime


# ── Context Schemas ───────────────────────────────────────────────────────────


class ContextResponse(BaseModel):
    """Context version metadata (content is encrypted, never in API response)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    version: int
    blob_size_bytes: int
    compression_ratio: float | None
    is_current: bool
    parent_version_id: UUID | None
    metadata: dict
    created_at: datetime


# ── Shift Schemas ─────────────────────────────────────────────────────────────


class ShiftRequest(BaseModel):
    """Request to shift a context to a different LLM."""

    source_context_id: UUID
    target_llm: str = Field(max_length=50)
    target_model: str = Field(default="", max_length=100)
    token_budget: int | None = Field(
        default=None,
        ge=1000,
        description="Maximum tokens for the target context. None = use model default.",
    )
    preserve_artifacts: bool = Field(
        default=True,
        description="Whether to include code artifacts in the shifted context.",
    )
    compression_level: str = Field(
        default="balanced",
        pattern="^(minimal|balanced|aggressive)$",
        description="How aggressively to compress. 'minimal' keeps more detail.",
    )


class ShiftResponse(BaseModel):
    """Result of a context shift operation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_llm: str
    target_llm: str
    source_tokens: int
    target_tokens: int
    compression_ratio: float | None
    processing_time_ms: int | None
    status: str
    created_at: datetime


# ── API Key Schemas ───────────────────────────────────────────────────────────


class APIKeyCreate(BaseModel):
    """Create a new API key."""

    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(
        default=["contexts:read", "contexts:write", "shifts:execute"],
    )


class APIKeyResponse(BaseModel):
    """API key metadata (never returns the full key after creation)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    key_prefix: str
    scopes: list
    is_active: bool
    last_used: datetime | None
    created_at: datetime


class APIKeyCreated(APIKeyResponse):
    """Returned ONLY at creation time — includes the full key.

    The full key is shown once and never stored. If the user loses it,
    they must create a new one. This is the security-standard approach
    used by Stripe, OpenAI, GitHub, and every serious API provider.
    """

    key: str = Field(description="Full API key. Shown only once at creation.")


# ── Pagination ────────────────────────────────────────────────────────────────


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Envelope for paginated list responses."""

    items: list
    total: int
    page: int
    limit: int
    has_more: bool


# ── Error Response ────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response format.

    Every error from the API follows this format, making it predictable
    for SDK/CLI consumers to handle errors programmatically.
    """

    error: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error description.")
    request_id: str = Field(description="Request ID for support correlation.")
    details: dict | None = Field(
        default=None,
        description="Additional error context (field-level validation errors, etc.).",
    )
