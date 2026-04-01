"""Default/strict/custom security policy tiers.

The Policy Engine defines what level of security each user/team gets.
Enterprise customers demand different policies than free-tier users:

    DEFAULT (free tier):
        - PII redaction: ON
        - Injection detection: ON
        - Encryption: AES-256-GCM
        - Retention: 30 days
        - Audit logging: basic

    STRICT (pro/team tier):
        - Everything in DEFAULT, plus:
        - Block on critical injections
        - Full audit trail
        - Retention: 90 days
        - IP allowlisting available

    ENTERPRISE (enterprise tier):
        - Everything in STRICT, plus:
        - Custom retention policies
        - Data residency controls
        - SOC2/HIPAA compliance mode
        - DLP (Data Loss Prevention) rules
        - Custom blocklists

Usage:
    from app.core.security.policy_engine import PolicyEngine, SecurityTier

    engine = PolicyEngine()
    policy = engine.get_policy(SecurityTier.STRICT)
    # policy.redact_pii == True
    # policy.block_on_critical == True
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SecurityTier(str, Enum):
    """Security tier levels."""

    DEFAULT = "default"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class SecurityPolicy:
    """A complete security policy configuration.

    Every security decision in the system checks this policy first.
    """

    tier: SecurityTier
    redact_pii: bool = True
    detect_injections: bool = True
    block_on_critical: bool = False
    encrypt_at_rest: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    retention_days: int = 30
    audit_logging: bool = True
    audit_detail_level: str = "basic"  # basic, detailed, full
    ip_allowlist: tuple[str, ...] = ()
    custom_blocklist: tuple[str, ...] = ()
    require_mfa: bool = False
    data_residency: str = ""  # e.g., "us", "eu", "" = any
    dlp_enabled: bool = False
    max_context_size_bytes: int = 10_000_000  # 10MB default


# -- Pre-built policies ------------------------------------------------------

_DEFAULT_POLICY = SecurityPolicy(
    tier=SecurityTier.DEFAULT,
    redact_pii=True,
    detect_injections=True,
    block_on_critical=False,
    encrypt_at_rest=True,
    retention_days=30,
    audit_logging=True,
    audit_detail_level="basic",
)

_STRICT_POLICY = SecurityPolicy(
    tier=SecurityTier.STRICT,
    redact_pii=True,
    detect_injections=True,
    block_on_critical=True,
    encrypt_at_rest=True,
    retention_days=90,
    audit_logging=True,
    audit_detail_level="detailed",
    require_mfa=True,
)

_ENTERPRISE_POLICY = SecurityPolicy(
    tier=SecurityTier.ENTERPRISE,
    redact_pii=True,
    detect_injections=True,
    block_on_critical=True,
    encrypt_at_rest=True,
    retention_days=365,
    audit_logging=True,
    audit_detail_level="full",
    require_mfa=True,
    dlp_enabled=True,
    max_context_size_bytes=50_000_000,  # 50MB for enterprise
)

_POLICIES: dict[SecurityTier, SecurityPolicy] = {
    SecurityTier.DEFAULT: _DEFAULT_POLICY,
    SecurityTier.STRICT: _STRICT_POLICY,
    SecurityTier.ENTERPRISE: _ENTERPRISE_POLICY,
}


# -- Policy Engine -----------------------------------------------------------


class PolicyEngine:
    """Resolves security policies for users and teams.

    Usage:
        engine = PolicyEngine()
        policy = engine.get_policy(SecurityTier.STRICT)

        # Or with custom overrides
        custom = engine.create_custom_policy(
            base_tier=SecurityTier.STRICT,
            retention_days=180,
            data_residency="eu",
        )
    """

    def get_policy(self, tier: SecurityTier) -> SecurityPolicy:
        """Get the pre-built policy for a security tier.

        Args:
            tier: Security tier level.

        Returns:
            SecurityPolicy for the tier.
        """
        return _POLICIES[tier]

    def get_policy_for_subscription(self, subscription_tier: str) -> SecurityPolicy:
        """Map a subscription tier to a security policy.

        Args:
            subscription_tier: User's subscription (free, pro, team, enterprise).

        Returns:
            Appropriate SecurityPolicy.
        """
        mapping: dict[str, SecurityTier] = {
            "free": SecurityTier.DEFAULT,
            "pro": SecurityTier.STRICT,
            "team": SecurityTier.STRICT,
            "enterprise": SecurityTier.ENTERPRISE,
        }
        security_tier = mapping.get(subscription_tier, SecurityTier.DEFAULT)
        return self.get_policy(security_tier)

    @staticmethod
    def create_custom_policy(
        base_tier: SecurityTier = SecurityTier.DEFAULT,
        **overrides: object,
    ) -> SecurityPolicy:
        """Create a custom policy based on a tier with overrides.

        Args:
            base_tier: Starting tier to customize.
            **overrides: Fields to override on the base policy.

        Returns:
            New SecurityPolicy with overrides applied.
        """
        base = _POLICIES[base_tier]
        base_dict = {
            "tier": base.tier,
            "redact_pii": base.redact_pii,
            "detect_injections": base.detect_injections,
            "block_on_critical": base.block_on_critical,
            "encrypt_at_rest": base.encrypt_at_rest,
            "encryption_algorithm": base.encryption_algorithm,
            "retention_days": base.retention_days,
            "audit_logging": base.audit_logging,
            "audit_detail_level": base.audit_detail_level,
            "ip_allowlist": base.ip_allowlist,
            "custom_blocklist": base.custom_blocklist,
            "require_mfa": base.require_mfa,
            "data_residency": base.data_residency,
            "dlp_enabled": base.dlp_enabled,
            "max_context_size_bytes": base.max_context_size_bytes,
        }
        base_dict.update(overrides)
        return SecurityPolicy(**base_dict)

    @staticmethod
    def validate_policy(policy: SecurityPolicy) -> list[str]:
        """Validate a security policy for consistency.

        Returns:
            List of warning messages (empty if valid).
        """
        warnings: list[str] = []

        if not policy.encrypt_at_rest:
            warnings.append("Encryption at rest is disabled -- not recommended")

        if policy.retention_days < 1:
            warnings.append("Retention period is less than 1 day")

        if policy.retention_days > 3650:
            warnings.append("Retention period exceeds 10 years")

        if policy.dlp_enabled and not policy.redact_pii:
            warnings.append("DLP is enabled but PII redaction is disabled")

        if policy.block_on_critical and not policy.detect_injections:
            warnings.append(
                "Block on critical is enabled but injection detection is disabled"
            )

        return warnings
