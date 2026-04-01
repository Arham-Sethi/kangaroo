"""Configuration system tests.

Tests cover:
    - Default values for local development
    - Environment variable parsing
    - Production secret validation (crash on defaults in prod)
    - S3 config validation
    - CORS origin parsing
    - Caching behavior
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError


class TestSettingsDefaults:
    """Test that defaults work for local development."""

    def test_default_environment(self):
        from app.config import Settings

        s = Settings()
        assert s.environment == "development"
        assert s.is_development is True
        assert s.is_production is False

    def test_default_database_url(self):
        from app.config import Settings

        s = Settings()
        assert "kangaroo" in s.database_url
        assert "asyncpg" in s.database_url

    def test_default_redis_url(self):
        from app.config import Settings

        s = Settings()
        assert s.redis_url == "redis://localhost:6379/0"

    def test_default_feature_flags(self):
        from app.config import Settings

        s = Settings()
        assert s.local_mode_enabled is True
        assert s.parallel_queries_enabled is True
        assert s.team_features_enabled is False

    def test_default_rate_limits(self):
        from app.config import Settings

        s = Settings()
        assert s.rate_limit_free_per_hour == 20
        assert s.rate_limit_pro_per_hour == 500
        assert s.rate_limit_team_per_hour == 2000


class TestProductionValidation:
    """Test that production mode rejects default secrets."""

    def test_production_rejects_default_secret_key(self):
        from app.config import Settings

        with pytest.raises(ValidationError, match="SECURITY"):
            Settings(environment="production")

    def test_production_rejects_default_jwt_secret(self):
        from app.config import Settings

        with pytest.raises(ValidationError, match="SECURITY"):
            Settings(
                environment="production",
                secret_key="a-real-secret-key-that-is-secure-enough",
            )

    def test_staging_also_validates(self):
        from app.config import Settings

        with pytest.raises(ValidationError, match="SECURITY"):
            Settings(environment="staging")

    def test_development_allows_defaults(self):
        from app.config import Settings

        s = Settings(environment="development")
        assert s.is_development is True


class TestS3Validation:
    """Test S3 configuration validation."""

    def test_s3_requires_bucket(self):
        from app.config import Settings

        with pytest.raises(ValidationError, match="S3_BUCKET"):
            Settings(storage_backend="s3")

    def test_s3_requires_region(self):
        from app.config import Settings

        with pytest.raises(ValidationError, match="S3_REGION"):
            Settings(storage_backend="s3", s3_bucket="my-bucket")

    def test_local_storage_no_s3_needed(self):
        from app.config import Settings

        s = Settings(storage_backend="local")
        assert s.storage_backend == "local"


class TestCORSParsing:
    """Test CORS origin parsing from environment."""

    def test_comma_separated_string(self):
        from app.config import Settings

        s = Settings(allowed_origins="http://a.com, http://b.com")
        assert s.allowed_origins == ["http://a.com", "http://b.com"]

    def test_list_input(self):
        from app.config import Settings

        s = Settings(allowed_origins=["http://a.com", "http://b.com"])
        assert len(s.allowed_origins) == 2

    def test_empty_string(self):
        from app.config import Settings

        s = Settings(allowed_origins="")
        assert s.allowed_origins == []


class TestCaching:
    """Test that get_settings() returns cached instance."""

    def test_same_instance(self):
        from app.config import get_settings

        # Clear cache
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear(self):
        from app.config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # Different instances after cache clear
        assert s1 is not s2
