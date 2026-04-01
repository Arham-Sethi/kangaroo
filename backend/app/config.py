"""Application configuration via environment variables.

Every setting in Kangaroo Shift flows through this module. Pydantic Settings
validates all values at startup — if a required secret is missing or malformed,
the application crashes immediately with a clear error rather than failing
silently at runtime when a customer hits that codepath.

Configuration categories:
    - Database: PostgreSQL connection, pool sizing
    - Redis: Cache and message queue connection
    - Meilisearch: Search index connection
    - Security: JWT, encryption, CORS
    - LLM: API keys for OpenAI, Anthropic, Google
    - Server: Host, port, debug, logging
    - Features: Feature flags for tier gating
    - Storage: Object storage backend (local/S3)
    - Rate Limiting: Per-tier request limits

Usage:
    from app.config import get_settings
    settings = get_settings()
    print(settings.database_url)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Kangaroo Shift backend.

    All values are read from environment variables. Defaults are set for
    local development — production deployments MUST override secrets.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    debug: bool = False
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    environment: Literal["development", "staging", "production"] = "development"
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins. Comma-separated in env var.",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: object) -> list[str]:
        """Parse comma-separated origins string into a list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return list(v) if isinstance(v, (list, tuple)) else []

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://kangaroo:kangaroo@localhost:5432/kangaroo",
        description="Async database URL (asyncpg driver).",
    )
    database_sync_url: str = Field(
        default="postgresql://kangaroo:kangaroo@localhost:5432/kangaroo",
        description="Sync database URL for Alembic migrations.",
    )
    db_pool_size: int = Field(default=20, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=50)
    db_pool_recycle: int = Field(
        default=3600,
        description="Seconds before a connection is recycled to prevent stale connections.",
    )

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = Field(default=50, ge=1)

    # ── Meilisearch ───────────────────────────────────────────────────────
    meili_url: str = "http://localhost:7700"
    meili_master_key: SecretStr = SecretStr("kangaroo-dev-master-key")

    # ── JWT Authentication ────────────────────────────────────────────────
    secret_key: SecretStr = SecretStr("change-me-in-production-use-a-long-random-string")
    jwt_secret_key: SecretStr = SecretStr("change-me-jwt-secret-key")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=30, ge=1)
    jwt_refresh_token_expire_days: int = Field(default=7, ge=1)

    # ── Encryption ────────────────────────────────────────────────────────
    encryption_key_derivation_memory: int = Field(
        default=65536,
        description="Argon2id memory cost in KiB (64 MB default).",
    )
    encryption_key_derivation_time: int = Field(
        default=3,
        description="Argon2id time cost (iterations).",
    )
    encryption_key_derivation_parallelism: int = Field(
        default=4,
        description="Argon2id parallelism degree.",
    )

    # ── LLM API Keys ─────────────────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # ── Feature Flags ─────────────────────────────────────────────────────
    local_mode_enabled: bool = True
    parallel_queries_enabled: bool = True
    team_features_enabled: bool = False

    # ── Object Storage ────────────────────────────────────────────────────
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_path: str = "./storage"
    s3_bucket: str | None = None
    s3_region: str | None = None

    # ── Rate Limiting ─────────────────────────────────────────────────────
    rate_limit_free_per_hour: int = Field(default=20, ge=1)
    rate_limit_pro_per_hour: int = Field(default=500, ge=1)
    rate_limit_team_per_hour: int = Field(default=2000, ge=1)

    # ── Validation ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Ensure production deployments have real secrets configured.

        In development, default secrets are acceptable for convenience.
        In staging/production, default secrets would be a critical
        security vulnerability — fail hard and fast.
        """
        if self.environment in ("staging", "production"):
            default_markers = ("change-me", "dev-", "kangaroo-dev")
            secret_fields = {
                "secret_key": self.secret_key,
                "jwt_secret_key": self.jwt_secret_key,
                "meili_master_key": self.meili_master_key,
            }
            for field_name, secret_value in secret_fields.items():
                value = secret_value.get_secret_value()
                if any(marker in value.lower() for marker in default_markers):
                    raise ValueError(
                        f"SECURITY: '{field_name}' uses a default/dev value in "
                        f"{self.environment} environment. Set a strong, unique secret."
                    )
        return self

    @model_validator(mode="after")
    def validate_s3_config(self) -> "Settings":
        """Ensure S3 config is complete when S3 backend is selected."""
        if self.storage_backend == "s3":
            if not self.s3_bucket:
                raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3")
            if not self.s3_region:
                raise ValueError("S3_REGION is required when STORAGE_BACKEND=s3")
        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings.

    Uses lru_cache so the Settings object is created once and reused.
    This avoids re-reading environment variables and re-validating on
    every request. The cache is process-level — each worker gets its own.
    """
    return Settings()
