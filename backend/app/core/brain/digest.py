"""Daily digest generation — "Here's what you learned yesterday."

Summarizes recent sessions into a structured digest highlighting:
    - New entities discovered
    - Decisions made or changed
    - Tasks created or completed
    - Key topics discussed

The digest is computed from a list of UCS documents within a time window.

Usage:
    generator = DigestGenerator()
    digest = generator.generate(recent_sessions, period_label="Yesterday")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    Entity,
    Task,
    TaskStatus,
    UniversalContextSchema,
)


@dataclass(frozen=True)
class DigestConfig:
    """Configuration for digest generation.

    Attributes:
        max_entities: Maximum entities to include in digest.
        max_decisions: Maximum decisions to include.
        max_tasks: Maximum tasks to include.
        min_entity_importance: Minimum importance to include an entity.
    """

    max_entities: int = 10
    max_decisions: int = 10
    max_tasks: int = 10
    min_entity_importance: float = 0.3


@dataclass(frozen=True)
class DigestEntry:
    """A single item in the digest.

    Attributes:
        category: Category (entity, decision, task, topic).
        title: Short title for the entry.
        detail: Longer description.
        importance: Relevance score (0..1).
        metadata: Additional context.
    """

    category: str
    title: str
    detail: str
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Digest:
    """A complete digest of recent activity.

    Attributes:
        period_label: Human-readable period (e.g., "Yesterday", "Last 7 days").
        session_count: Number of sessions in the digest period.
        entries: Ordered list of digest entries.
        new_entities: Count of new entities discovered.
        decisions_made: Count of decisions made.
        tasks_completed: Count of tasks completed.
        total_messages: Total messages across sessions.
    """

    period_label: str
    session_count: int
    entries: tuple[DigestEntry, ...]
    new_entities: int
    decisions_made: int
    tasks_completed: int
    total_messages: int


class DigestGenerator:
    """Generates structured digests from recent UCS sessions.

    The generator collects entities, decisions, and tasks from
    the given sessions, ranks them by importance, and produces
    a digestible summary.
    """

    def __init__(self, config: DigestConfig | None = None) -> None:
        self._config = config or DigestConfig()

    @property
    def config(self) -> DigestConfig:
        return self._config

    def generate(
        self,
        sessions: list[UniversalContextSchema],
        period_label: str = "Recent",
    ) -> Digest:
        """Generate a digest from recent sessions.

        Args:
            sessions: UCS documents from the digest period.
            period_label: Human-readable label for the period.

        Returns:
            A Digest summarizing the sessions.
        """
        if not sessions:
            return Digest(
                period_label=period_label,
                session_count=0,
                entries=(),
                new_entities=0,
                decisions_made=0,
                tasks_completed=0,
                total_messages=0,
            )

        entries: list[DigestEntry] = []

        # Collect entities
        entity_entries = self._collect_entities(sessions)
        entries.extend(entity_entries)

        # Collect decisions
        decision_entries = self._collect_decisions(sessions)
        entries.extend(decision_entries)

        # Collect tasks
        task_entries = self._collect_tasks(sessions)
        entries.extend(task_entries)

        # Sort all entries by importance descending
        entries.sort(key=lambda e: -e.importance)

        # Count stats
        all_entities = self._all_entities(sessions)
        all_decisions = self._all_active_decisions(sessions)
        completed_tasks = self._completed_tasks(sessions)
        total_messages = sum(s.session_meta.message_count for s in sessions)

        return Digest(
            period_label=period_label,
            session_count=len(sessions),
            entries=tuple(entries),
            new_entities=len(all_entities),
            decisions_made=len(all_decisions),
            tasks_completed=len(completed_tasks),
            total_messages=total_messages,
        )

    def _collect_entities(
        self, sessions: list[UniversalContextSchema]
    ) -> list[DigestEntry]:
        """Extract notable entities from sessions."""
        cfg = self._config
        seen: dict[str, Entity] = {}

        for ucs in sessions:
            for entity in ucs.entities:
                key = entity.name.lower().strip()
                if key not in seen or entity.importance > seen[key].importance:
                    seen[key] = entity

        # Filter by importance and limit
        entities = sorted(seen.values(), key=lambda e: -e.importance)
        entities = [
            e for e in entities if e.importance >= cfg.min_entity_importance
        ]
        entities = entities[: cfg.max_entities]

        return [
            DigestEntry(
                category="entity",
                title=e.name,
                detail=f"{e.type.value} entity (importance: {e.importance:.2f})",
                importance=e.importance,
                metadata={"entity_type": e.type.value},
            )
            for e in entities
        ]

    def _collect_decisions(
        self, sessions: list[UniversalContextSchema]
    ) -> list[DigestEntry]:
        """Extract decisions from sessions."""
        cfg = self._config
        seen: dict[str, Decision] = {}

        for ucs in sessions:
            for decision in ucs.decisions:
                if decision.status != DecisionStatus.ACTIVE:
                    continue
                key = decision.description.lower().strip()
                if key not in seen:
                    seen[key] = decision

        decisions = list(seen.values())[: cfg.max_decisions]

        return [
            DigestEntry(
                category="decision",
                title=d.description[:80],
                detail=d.rationale or "No rationale provided",
                importance=0.7,
                metadata={"status": d.status.value},
            )
            for d in decisions
        ]

    def _collect_tasks(
        self, sessions: list[UniversalContextSchema]
    ) -> list[DigestEntry]:
        """Extract task updates from sessions."""
        cfg = self._config
        seen: dict[str, Task] = {}

        for ucs in sessions:
            for task in ucs.tasks:
                key = task.description.lower().strip()
                if key not in seen:
                    seen[key] = task
                elif task.status == TaskStatus.COMPLETED:
                    seen[key] = task

        tasks = list(seen.values())[: cfg.max_tasks]

        return [
            DigestEntry(
                category="task",
                title=t.description[:80],
                detail=f"Status: {t.status.value}, Priority: {t.priority:.1f}",
                importance=t.priority,
                metadata={"status": t.status.value, "priority": t.priority},
            )
            for t in tasks
        ]

    def _all_entities(
        self, sessions: list[UniversalContextSchema]
    ) -> list[Entity]:
        """Deduplicated entities from all sessions."""
        seen: dict[str, Entity] = {}
        for ucs in sessions:
            for e in ucs.entities:
                key = e.name.lower().strip()
                if key not in seen:
                    seen[key] = e
        return list(seen.values())

    def _all_active_decisions(
        self, sessions: list[UniversalContextSchema]
    ) -> list[Decision]:
        """Active decisions from all sessions."""
        seen: dict[str, Decision] = {}
        for ucs in sessions:
            for d in ucs.decisions:
                if d.status == DecisionStatus.ACTIVE:
                    key = d.description.lower().strip()
                    if key not in seen:
                        seen[key] = d
        return list(seen.values())

    def _completed_tasks(
        self, sessions: list[UniversalContextSchema]
    ) -> list[Task]:
        """Completed tasks from all sessions."""
        result = []
        for ucs in sessions:
            for t in ucs.tasks:
                if t.status == TaskStatus.COMPLETED:
                    result.append(t)
        return result
