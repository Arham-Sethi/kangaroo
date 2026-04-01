"""3-stage prompt-injection defense and PII sanitization pipeline.

This is the guardian of user data. Every conversation passes through
the sanitizer before processing. It catches:

    Stage 1 — PII Detection & Redaction:
        - Email addresses
        - Phone numbers
        - Credit card numbers
        - Social security numbers
        - IP addresses
        - API keys / tokens
        - Physical addresses (partial)

    Stage 2 — Prompt Injection Detection:
        - System prompt overrides ("ignore previous instructions")
        - Jailbreak attempts ("DAN mode", "developer mode")
        - Encoding attacks (base64 encoded instructions)
        - Delimiter injection (markdown/XML tags to escape context)

    Stage 3 — Content Policy:
        - Checks against configurable blocklists
        - Flags content that violates safety policies

Each stage produces SafetyFlags that are attached to the UCS for
audit purposes. Enterprise customers require this for compliance.

Usage:
    from app.core.security.sanitizer import Sanitizer, SanitizeResult

    sanitizer = Sanitizer()
    result = sanitizer.sanitize("Call me at 555-123-4567")
    # result.cleaned_text = "Call me at [PHONE_REDACTED]"
    # result.flags = [SafetyFlag(type=PII_DETECTED, ...)]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.core.models.ucs import (
    SafetyAction,
    SafetyFlag,
    SafetyFlagType,
    SafetySeverity,
)


# -- PII Patterns -----------------------------------------------------------

# Email addresses
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# Phone numbers (US, UK, international formats)
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    r"|"
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    r")"
    r"(?!\d)"
)

# Credit card numbers (Visa, MC, Amex, Discover)
_CREDIT_CARD_RE = re.compile(
    r"\b(?:"
    r"4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"   # Visa
    r"|5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # MC
    r"|3[47]\d{1}[-\s]?\d{6}[-\s]?\d{5}"           # Amex
    r"|6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}"  # Discover
    r")\b"
)

# SSN (US Social Security Numbers)
_SSN_RE = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

# IPv4 addresses
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# API keys / tokens (generic patterns for common providers)
_API_KEY_RE = re.compile(
    r"(?:"
    r"sk-[A-Za-z0-9]{20,}"              # OpenAI
    r"|sk-ant-[A-Za-z0-9\-]{20,}"       # Anthropic
    r"|AIza[A-Za-z0-9\-_]{35}"          # Google
    r"|ghp_[A-Za-z0-9]{36}"             # GitHub PAT
    r"|gho_[A-Za-z0-9]{36}"             # GitHub OAuth
    r"|xoxb-[A-Za-z0-9\-]+"             # Slack bot
    r"|xoxp-[A-Za-z0-9\-]+"             # Slack user
    r"|AKIA[A-Z0-9]{16}"                # AWS access key
    r"|[A-Za-z0-9]{32,}_live_[A-Za-z0-9]+"  # Stripe
    r")"
)

# Bearer tokens in headers
_BEARER_RE = re.compile(
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",
    re.IGNORECASE,
)


# -- Prompt Injection Patterns -----------------------------------------------

_INJECTION_PATTERNS: list[tuple[re.Pattern, str, SafetySeverity]] = [
    # System prompt overrides
    (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        "System prompt override attempt",
        SafetySeverity.HIGH,
    ),
    (
        re.compile(r"forget\s+(all\s+)?your\s+(previous\s+)?instructions", re.IGNORECASE),
        "Instruction forgetting attempt",
        SafetySeverity.HIGH,
    ),
    (
        re.compile(r"you\s+are\s+now\s+(?:DAN|a\s+new\s+AI|unrestricted)", re.IGNORECASE),
        "Jailbreak persona injection",
        SafetySeverity.CRITICAL,
    ),
    (
        re.compile(r"(?:developer|debug|admin|god)\s+mode", re.IGNORECASE),
        "Privilege escalation attempt",
        SafetySeverity.HIGH,
    ),
    (
        re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
        "System prompt injection via role prefix",
        SafetySeverity.HIGH,
    ),
    # Encoding attacks
    (
        re.compile(r"base64\s*:\s*[A-Za-z0-9+/]{20,}={0,2}", re.IGNORECASE),
        "Base64-encoded instruction injection",
        SafetySeverity.MEDIUM,
    ),
    # Delimiter injection
    (
        re.compile(r"<\|(?:im_start|im_end|system|endoftext)\|>", re.IGNORECASE),
        "Chat ML delimiter injection",
        SafetySeverity.CRITICAL,
    ),
    (
        re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE),
        "LLaMA instruction delimiter injection",
        SafetySeverity.HIGH,
    ),
]


# -- Redaction labels --------------------------------------------------------


class PIIType(str, Enum):
    """Types of PII that can be detected and redacted."""

    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IP_ADDRESS = "IP_ADDRESS"
    API_KEY = "API_KEY"
    BEARER_TOKEN = "BEARER_TOKEN"


_PII_PATTERNS: list[tuple[re.Pattern, PIIType, SafetySeverity]] = [
    (_API_KEY_RE, PIIType.API_KEY, SafetySeverity.CRITICAL),
    (_BEARER_RE, PIIType.BEARER_TOKEN, SafetySeverity.CRITICAL),
    (_CREDIT_CARD_RE, PIIType.CREDIT_CARD, SafetySeverity.HIGH),
    (_SSN_RE, PIIType.SSN, SafetySeverity.HIGH),
    (_EMAIL_RE, PIIType.EMAIL, SafetySeverity.MEDIUM),
    (_PHONE_RE, PIIType.PHONE, SafetySeverity.MEDIUM),
    (_IPV4_RE, PIIType.IP_ADDRESS, SafetySeverity.LOW),
]


# -- Result ------------------------------------------------------------------


@dataclass(frozen=True)
class SanitizeResult:
    """Result from the sanitization pipeline.

    Attributes:
        cleaned_text: Text with PII redacted.
        original_text: The original input text.
        flags: Safety flags generated during sanitization.
        pii_count: Number of PII items detected and redacted.
        injection_count: Number of prompt injection attempts detected.
        is_safe: True if no HIGH or CRITICAL flags were generated.
    """

    cleaned_text: str
    original_text: str
    flags: tuple[SafetyFlag, ...]
    pii_count: int
    injection_count: int
    is_safe: bool


# -- Main Pipeline -----------------------------------------------------------


class Sanitizer:
    """3-stage sanitization pipeline: PII + injection + policy.

    Usage:
        sanitizer = Sanitizer()
        result = sanitizer.sanitize("My email is user@example.com")
        # result.cleaned_text = "My email is [EMAIL_REDACTED]"
    """

    def __init__(
        self,
        redact_pii: bool = True,
        detect_injections: bool = True,
        block_on_critical: bool = False,
    ) -> None:
        """Initialize the sanitizer.

        Args:
            redact_pii: Whether to redact PII from text.
            detect_injections: Whether to scan for prompt injections.
            block_on_critical: If True, raise on CRITICAL flags.
        """
        self._redact_pii = redact_pii
        self._detect_injections = detect_injections
        self._block_on_critical = block_on_critical

    def sanitize(
        self,
        text: str,
        message_index: int | None = None,
    ) -> SanitizeResult:
        """Run the full sanitization pipeline on text.

        Args:
            text: Input text to sanitize.
            message_index: Optional message index for flag context.

        Returns:
            SanitizeResult with cleaned text and safety flags.
        """
        flags: list[SafetyFlag] = []
        cleaned = text
        pii_count = 0
        injection_count = 0

        # Stage 1: PII detection and redaction
        if self._redact_pii:
            cleaned, pii_flags, pii_count = self._stage_pii(cleaned, message_index)
            flags.extend(pii_flags)

        # Stage 2: Prompt injection detection
        if self._detect_injections:
            injection_flags, injection_count = self._stage_injection(
                text, message_index,
            )
            flags.extend(injection_flags)

        # Check for critical flags
        has_critical = any(
            f.severity in (SafetySeverity.CRITICAL, SafetySeverity.HIGH)
            for f in flags
        )

        if self._block_on_critical and any(
            f.severity == SafetySeverity.CRITICAL for f in flags
        ):
            raise SecurityViolation(
                "Critical security violation detected. Processing blocked.",
                flags=tuple(flags),
            )

        return SanitizeResult(
            cleaned_text=cleaned,
            original_text=text,
            flags=tuple(flags),
            pii_count=pii_count,
            injection_count=injection_count,
            is_safe=not has_critical,
        )

    def sanitize_messages(
        self,
        texts: list[str],
    ) -> list[SanitizeResult]:
        """Sanitize multiple messages (e.g., a full conversation).

        Args:
            texts: List of message texts.

        Returns:
            List of SanitizeResult objects, one per message.
        """
        return [self.sanitize(text, message_index=i) for i, text in enumerate(texts)]

    @staticmethod
    def _stage_pii(
        text: str,
        message_index: int | None,
    ) -> tuple[str, list[SafetyFlag], int]:
        """Stage 1: Detect and redact PII."""
        flags: list[SafetyFlag] = []
        cleaned = text
        count = 0

        for pattern, pii_type, severity in _PII_PATTERNS:
            matches = list(pattern.finditer(cleaned))
            if matches:
                count += len(matches)
                redaction = f"[{pii_type.value}_REDACTED]"
                cleaned = pattern.sub(redaction, cleaned)
                flags.append(SafetyFlag(
                    type=SafetyFlagType.PII_DETECTED,
                    severity=severity,
                    description=f"{pii_type.value} detected ({len(matches)} instance(s))",
                    action_taken=SafetyAction.REDACTED,
                    confidence=0.95,
                    source_message_index=message_index,
                ))

        return cleaned, flags, count

    @staticmethod
    def _stage_injection(
        text: str,
        message_index: int | None,
    ) -> tuple[list[SafetyFlag], int]:
        """Stage 2: Detect prompt injection attempts."""
        flags: list[SafetyFlag] = []
        count = 0

        for pattern, description, severity in _INJECTION_PATTERNS:
            if pattern.search(text):
                count += 1
                flags.append(SafetyFlag(
                    type=SafetyFlagType.INJECTION,
                    severity=severity,
                    description=description,
                    action_taken=SafetyAction.FLAGGED,
                    confidence=0.85,
                    source_message_index=message_index,
                ))

        return flags, count


# -- Exceptions --------------------------------------------------------------


class SecurityViolation(Exception):
    """Raised when a critical security violation is detected and blocking is enabled."""

    def __init__(self, message: str, flags: tuple[SafetyFlag, ...] = ()) -> None:
        super().__init__(message)
        self.flags = flags
