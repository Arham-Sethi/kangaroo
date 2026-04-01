"""Base adapter interface for LLM output reconstruction.

Adapters are the OUTPUT side of Kangaroo Shift. While parsers convert
LLM-specific formats INTO the universal CCR/UCS, adapters convert
UCS back OUT into LLM-specific formats.

The flow:
    ChatGPT conversation -> Parser -> CCR -> Engine -> UCS
    UCS -> Adapter -> Claude format -> Ready for Claude API

Every adapter implements the same interface:
    adapt(ucs) -> dict   (LLM-specific message format)
    format_name -> str    (e.g., "openai", "claude", "gemini")

Usage:
    from app.core.adapters import AdapterRegistry

    registry = AdapterRegistry()
    output = registry.adapt(ucs, target="claude")
    # output is a dict ready for the Claude API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.models.ucs import UniversalContextSchema


@dataclass(frozen=True)
class AdaptedOutput:
    """Result from an adapter conversion.

    Attributes:
        format_name: Target LLM format (openai, claude, gemini).
        messages: List of messages in the target format.
        system_prompt: Reconstructed system prompt (if applicable).
        metadata: Additional format-specific metadata.
        token_estimate: Estimated token count of the output.
    """

    format_name: str
    messages: list[dict[str, Any]]
    system_prompt: str
    metadata: dict[str, Any]
    token_estimate: int


class BaseAdapter(ABC):
    """Abstract base class for all output adapters.

    Subclasses must implement:
        - format_name: The target format identifier
        - adapt(): Convert UCS to target format
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format identifier (e.g., 'openai', 'claude')."""
        ...

    @abstractmethod
    def adapt(self, ucs: UniversalContextSchema) -> AdaptedOutput:
        """Convert a UCS into the target LLM format.

        Args:
            ucs: Universal Context Schema to convert.

        Returns:
            AdaptedOutput with messages in the target format.
        """
        ...

    def _build_context_summary(self, ucs: UniversalContextSchema) -> str:
        """Build a human-readable context summary from UCS components.

        This is injected into the system prompt to give the target LLM
        full awareness of the prior conversation's context.
        """
        parts: list[str] = []

        # Global summary
        from app.core.models.ucs import SummaryLevel
        global_summaries = [s for s in ucs.summaries if s.level == SummaryLevel.GLOBAL]
        if global_summaries:
            parts.append("## Previous Conversation Summary")
            parts.append(global_summaries[0].content)

        # Topic summaries
        topic_summaries = [s for s in ucs.summaries if s.level == SummaryLevel.TOPIC]
        if topic_summaries:
            parts.append("\n## Key Topics")
            for ts in topic_summaries:
                parts.append(f"- {ts.content[:200]}")

        # Key entities
        if ucs.entities:
            sorted_entities = sorted(
                ucs.entities,
                key=lambda e: e.importance,
                reverse=True,
            )[:15]  # Top 15 entities
            parts.append("\n## Key Entities")
            for entity in sorted_entities:
                aliases = f" (aka {', '.join(entity.aliases)})" if entity.aliases else ""
                parts.append(f"- **{entity.name}**{aliases} [{entity.type.value}]")

        # Active decisions
        from app.core.models.ucs import DecisionStatus
        active_decisions = [d for d in ucs.decisions if d.status == DecisionStatus.ACTIVE]
        if active_decisions:
            parts.append("\n## Active Decisions")
            for d in active_decisions:
                parts.append(f"- {d.description}")
                if d.rationale:
                    parts.append(f"  Rationale: {d.rationale[:200]}")

        # Active tasks
        from app.core.models.ucs import TaskStatus
        active_tasks = [t for t in ucs.tasks if t.status == TaskStatus.ACTIVE]
        if active_tasks:
            parts.append("\n## Open Tasks")
            for t in active_tasks:
                parts.append(f"- {t.description}")

        # Artifacts
        if ucs.artifacts:
            parts.append(f"\n## Artifacts ({len(ucs.artifacts)} items)")
            for a in ucs.artifacts[:5]:
                title = a.title or f"{a.type.value} ({a.language})"
                parts.append(f"- {title}")

        # Preferences
        prefs = ucs.preferences
        parts.append(f"\n## User Preferences")
        parts.append(f"- Tone: {prefs.tone.value}")
        parts.append(f"- Detail: {prefs.detail_level.value}")
        if prefs.domain_expertise:
            parts.append(f"- Expertise: {', '.join(prefs.domain_expertise)}")

        return "\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return max(1, len(text) // 4) if text else 0


class AdapterRegistry:
    """Registry for output adapters. Routes UCS to the correct format.

    Usage:
        registry = AdapterRegistry()
        output = registry.adapt(ucs, target="claude")
    """

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter) -> None:
        """Register an adapter."""
        self._adapters[adapter.format_name] = adapter

    @property
    def available_formats(self) -> list[str]:
        """List all registered format names."""
        return sorted(self._adapters.keys())

    def get_adapter(self, format_name: str) -> BaseAdapter | None:
        """Get an adapter by format name."""
        return self._adapters.get(format_name)

    def adapt(self, ucs: UniversalContextSchema, target: str) -> AdaptedOutput:
        """Convert UCS to a target format.

        Args:
            ucs: The Universal Context Schema.
            target: Target format name (openai, claude, gemini).

        Returns:
            AdaptedOutput with messages in the target format.

        Raises:
            ValueError: If the target format is not registered.
        """
        adapter = self._adapters.get(target)
        if adapter is None:
            available = ", ".join(self.available_formats)
            raise ValueError(
                f"Unknown target format '{target}'. Available: {available}"
            )
        return adapter.adapt(ucs)


def create_default_adapter_registry() -> AdapterRegistry:
    """Create an adapter registry with all built-in adapters.

    Returns:
        AdapterRegistry with OpenAI, Claude, and Gemini adapters.
    """
    from app.core.adapters.openai_adapter import OpenAIAdapter
    from app.core.adapters.claude_adapter import ClaudeAdapter
    from app.core.adapters.gemini_adapter import GeminiAdapter

    registry = AdapterRegistry()
    registry.register(OpenAIAdapter())
    registry.register(ClaudeAdapter())
    registry.register(GeminiAdapter())
    return registry
