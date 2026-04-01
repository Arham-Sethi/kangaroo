"""Tests for security policy engine.

Tests cover:
    - Pre-built policy tiers (DEFAULT, STRICT, ENTERPRISE)
    - Subscription tier mapping
    - Custom policy creation with overrides
    - Policy validation warnings
    - SecurityPolicy properties
"""

import pytest

from app.core.security.policy_engine import (
    PolicyEngine,
    SecurityPolicy,
    SecurityTier,
)


# == Pre-built Policies ======================================================


class TestPrebuiltPolicies:
    """Tests for default policy configurations."""

    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    def test_default_policy(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy(SecurityTier.DEFAULT)
        assert policy.tier == SecurityTier.DEFAULT
        assert policy.redact_pii is True
        assert policy.detect_injections is True
        assert policy.block_on_critical is False
        assert policy.encrypt_at_rest is True
        assert policy.retention_days == 30
        assert policy.audit_detail_level == "basic"
        assert policy.require_mfa is False

    def test_strict_policy(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy(SecurityTier.STRICT)
        assert policy.tier == SecurityTier.STRICT
        assert policy.block_on_critical is True
        assert policy.retention_days == 90
        assert policy.audit_detail_level == "detailed"
        assert policy.require_mfa is True

    def test_enterprise_policy(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy(SecurityTier.ENTERPRISE)
        assert policy.tier == SecurityTier.ENTERPRISE
        assert policy.retention_days == 365
        assert policy.audit_detail_level == "full"
        assert policy.require_mfa is True
        assert policy.dlp_enabled is True
        assert policy.max_context_size_bytes == 50_000_000

    def test_all_policies_encrypt_at_rest(self, engine: PolicyEngine) -> None:
        for tier in SecurityTier:
            policy = engine.get_policy(tier)
            assert policy.encrypt_at_rest is True

    def test_all_policies_redact_pii(self, engine: PolicyEngine) -> None:
        for tier in SecurityTier:
            policy = engine.get_policy(tier)
            assert policy.redact_pii is True

    def test_all_policies_detect_injections(self, engine: PolicyEngine) -> None:
        for tier in SecurityTier:
            policy = engine.get_policy(tier)
            assert policy.detect_injections is True


# == Subscription Mapping ====================================================


class TestSubscriptionMapping:
    """Tests for subscription tier to security policy mapping."""

    @pytest.fixture
    def engine(self) -> PolicyEngine:
        return PolicyEngine()

    def test_free_maps_to_default(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("free")
        assert policy.tier == SecurityTier.DEFAULT

    def test_pro_maps_to_strict(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("pro")
        assert policy.tier == SecurityTier.STRICT

    def test_team_maps_to_strict(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("team")
        assert policy.tier == SecurityTier.STRICT

    def test_enterprise_maps_to_enterprise(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("enterprise")
        assert policy.tier == SecurityTier.ENTERPRISE

    def test_unknown_subscription_defaults(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("nonexistent")
        assert policy.tier == SecurityTier.DEFAULT

    def test_empty_subscription_defaults(self, engine: PolicyEngine) -> None:
        policy = engine.get_policy_for_subscription("")
        assert policy.tier == SecurityTier.DEFAULT


# == Custom Policies =========================================================


class TestCustomPolicies:
    """Tests for custom policy creation."""

    def test_create_custom_from_default(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            base_tier=SecurityTier.DEFAULT,
            retention_days=180,
        )
        assert policy.retention_days == 180
        assert policy.tier == SecurityTier.DEFAULT

    def test_create_custom_from_strict(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            base_tier=SecurityTier.STRICT,
            data_residency="eu",
        )
        assert policy.data_residency == "eu"
        assert policy.block_on_critical is True  # inherited from STRICT

    def test_custom_preserves_base_values(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            base_tier=SecurityTier.ENTERPRISE,
            retention_days=730,
        )
        assert policy.dlp_enabled is True  # inherited
        assert policy.require_mfa is True  # inherited
        assert policy.retention_days == 730  # overridden

    def test_custom_with_multiple_overrides(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            base_tier=SecurityTier.DEFAULT,
            retention_days=60,
            data_residency="us",
            dlp_enabled=True,
            require_mfa=True,
            ip_allowlist=("10.0.0.0/8",),
        )
        assert policy.retention_days == 60
        assert policy.data_residency == "us"
        assert policy.dlp_enabled is True
        assert policy.require_mfa is True
        assert policy.ip_allowlist == ("10.0.0.0/8",)

    def test_custom_with_no_overrides_matches_base(self) -> None:
        base = PolicyEngine().get_policy(SecurityTier.STRICT)
        custom = PolicyEngine.create_custom_policy(base_tier=SecurityTier.STRICT)
        assert custom.retention_days == base.retention_days
        assert custom.block_on_critical == base.block_on_critical


# == Policy Validation =======================================================


class TestPolicyValidation:
    """Tests for policy consistency validation."""

    def test_valid_policy_no_warnings(self) -> None:
        policy = PolicyEngine().get_policy(SecurityTier.DEFAULT)
        warnings = PolicyEngine.validate_policy(policy)
        assert warnings == []

    def test_encryption_disabled_warning(self) -> None:
        policy = PolicyEngine.create_custom_policy(encrypt_at_rest=False)
        warnings = PolicyEngine.validate_policy(policy)
        assert any("Encryption at rest is disabled" in w for w in warnings)

    def test_retention_too_short_warning(self) -> None:
        policy = PolicyEngine.create_custom_policy(retention_days=0)
        warnings = PolicyEngine.validate_policy(policy)
        assert any("less than 1 day" in w for w in warnings)

    def test_retention_too_long_warning(self) -> None:
        policy = PolicyEngine.create_custom_policy(retention_days=5000)
        warnings = PolicyEngine.validate_policy(policy)
        assert any("exceeds 10 years" in w for w in warnings)

    def test_dlp_without_pii_warning(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            dlp_enabled=True,
            redact_pii=False,
        )
        warnings = PolicyEngine.validate_policy(policy)
        assert any("DLP is enabled but PII redaction is disabled" in w for w in warnings)

    def test_block_without_detection_warning(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            block_on_critical=True,
            detect_injections=False,
        )
        warnings = PolicyEngine.validate_policy(policy)
        assert any("injection detection is disabled" in w for w in warnings)

    def test_multiple_warnings(self) -> None:
        policy = PolicyEngine.create_custom_policy(
            encrypt_at_rest=False,
            retention_days=0,
            dlp_enabled=True,
            redact_pii=False,
        )
        warnings = PolicyEngine.validate_policy(policy)
        assert len(warnings) >= 3

    def test_all_prebuilt_policies_are_valid(self) -> None:
        engine = PolicyEngine()
        for tier in SecurityTier:
            warnings = PolicyEngine.validate_policy(engine.get_policy(tier))
            assert warnings == [], f"{tier} policy has warnings: {warnings}"


# == SecurityPolicy Properties ================================================


class TestSecurityPolicy:
    """Tests for SecurityPolicy dataclass."""

    def test_policy_is_frozen(self) -> None:
        policy = SecurityPolicy(tier=SecurityTier.DEFAULT)
        with pytest.raises(AttributeError):
            policy.retention_days = 999  # type: ignore[misc]

    def test_default_values(self) -> None:
        policy = SecurityPolicy(tier=SecurityTier.DEFAULT)
        assert policy.redact_pii is True
        assert policy.encryption_algorithm == "AES-256-GCM"
        assert policy.ip_allowlist == ()
        assert policy.custom_blocklist == ()
        assert policy.data_residency == ""
        assert policy.max_context_size_bytes == 10_000_000


# == SecurityTier Enum ========================================================


class TestSecurityTier:
    """Tests for tier enum."""

    def test_tier_values(self) -> None:
        assert SecurityTier.DEFAULT.value == "default"
        assert SecurityTier.STRICT.value == "strict"
        assert SecurityTier.ENTERPRISE.value == "enterprise"

    def test_tier_from_string(self) -> None:
        assert SecurityTier("default") == SecurityTier.DEFAULT
        assert SecurityTier("strict") == SecurityTier.STRICT
