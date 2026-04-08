"""Knowledge gap detection — find unresolved topics and missing decisions.

Analyzes a set of UCS sessions to identify:
    - Topics discussed multiple times without a decision
    - Stalled tasks (active for too long)
    - Entities mentioned frequently but with low importance (unclear)
    - Decisions that contradict without resolution

"You've been discussing auth for 2 weeks but haven't decided on session management."

Usage:
    detector = GapDetector()
    gaps = detector.detect(sessions, session_ages_days=[0, 7, 14])
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
class GapConfig:
    """Configuration for gap detection.

    Attributes:
        min_mentions_for_undecided: How many sessions must mention a topic
            before it's flagged as needing a decision.
        stalled_task_age_days: Days after which an active task is "stalled".
        low_importance_threshold: Entities below this are unclear/vague.
        min_entity_mentions: Minimum session mentions for unclear entity flag.
    """

    min_mentions_for_undecided: int = 2
    stalled_task_age_days: float = 14.0
    low_importance_threshold: float = 0.3
    min_entity_mentions: int = 3


@dataclass(frozen=True)
class KnowledgeGap:
    """A detected gap in the user's knowledge graph.

    Attributes:
        gap_type: Type of gap (undecided_topic, stalled_task, unclear_entity).
        title: Short description of the gap.
        detail: Longer explanation with context.
        severity: How important it is to address (high, medium, low).
        session_count: Number of sessions where this gap appears.
        metadata: Additional context.
    """

    gap_type: str
    title: str
    detail: str
    severity: str = "medium"
    session_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GapReport:
    """Complete gap analysis report.

    Attributes:
        gaps: List of detected knowledge gaps.
        sessions_analyzed: Number of sessions analyzed.
        total_entities: Total unique entities across sessions.
        total_decisions: Total unique decisions across sessions.
        total_tasks: Total unique tasks across sessions.
    """

    gaps: tuple[KnowledgeGap, ...]
    sessions_analyzed: int
    total_entities: int
    total_decisions: int
    total_tasks: int


class GapDetector:
    """Detects knowledge gaps across multiple UCS sessions.

    Scans for patterns that indicate incomplete thinking:
        1. Topics discussed repeatedly without decisions
        2. Tasks that have been active too long
        3. Entities mentioned often but poorly defined
    """

    def __init__(self, config: GapConfig | None = None) -> None:
        self._config = config or GapConfig()

    @property
    def config(self) -> GapConfig:
        return self._config

    def detect(
        self,
        sessions: list[UniversalContextSchema],
        session_ages_days: list[float] | None = None,
    ) -> GapReport:
        """Detect knowledge gaps across sessions.

        Args:
            sessions: UCS documents to analyze.
            session_ages_days: Age in days for each session.
                              If None, all treated as current.

        Returns:
            GapReport with detected gaps.
        """
        if not sessions:
            return GapReport(
                gaps=(),
                sessions_analyzed=0,
                total_entities=0,
                total_decisions=0,
                total_tasks=0,
            )

        ages = session_ages_days or [0.0] * len(sessions)
        gaps: list[KnowledgeGap] = []

        # Detection 1: Undecided topics
        gaps.extend(self._find_undecided_topics(sessions))

        # Detection 2: Stalled tasks
        gaps.extend(self._find_stalled_tasks(sessions, ages))

        # Detection 3: Unclear entities
        gaps.extend(self._find_unclear_entities(sessions))

        # Sort by severity (high > medium > low)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        gaps.sort(key=lambda g: severity_order.get(g.severity, 99))

        # Compute stats
        unique_entities = self._count_unique_entities(sessions)
        unique_decisions = self._count_unique_decisions(sessions)
        unique_tasks = self._count_unique_tasks(sessions)

        return GapReport(
            gaps=tuple(gaps),
            sessions_analyzed=len(sessions),
            total_entities=unique_entities,
            total_decisions=unique_decisions,
            total_tasks=unique_tasks,
        )

    def _find_undecided_topics(
        self, sessions: list[UniversalContextSchema]
    ) -> list[KnowledgeGap]:
        """Find entities discussed across multiple sessions without decisions."""
        cfg = self._config

        # Count sessions per entity
        entity_sessions: dict[str, int] = {}
        for ucs in sessions:
            seen_in_session: set[str] = set()
            for entity in ucs.entities:
                key = entity.name.lower().strip()
                if key not in seen_in_session:
                    entity_sessions[key] = entity_sessions.get(key, 0) + 1
                    seen_in_session.add(key)

        # Collect all decision topics
        decision_topics: set[str] = set()
        for ucs in sessions:
            for decision in ucs.decisions:
                # Extract entity names mentioned in decisions
                desc_lower = decision.description.lower()
                for entity_key in entity_sessions:
                    if entity_key in desc_lower:
                        decision_topics.add(entity_key)

        # Find entities discussed often but without decisions
        gaps: list[KnowledgeGap] = []
        for entity_key, count in entity_sessions.items():
            if (
                count >= cfg.min_mentions_for_undecided
                and entity_key not in decision_topics
            ):
                gaps.append(
                    KnowledgeGap(
                        gap_type="undecided_topic",
                        title=f"No decision made about '{entity_key}'",
                        detail=(
                            f"'{entity_key}' has been discussed in {count} sessions "
                            f"but no decision has been recorded about it."
                        ),
                        severity="high" if count >= 3 else "medium",
                        session_count=count,
                        metadata={"entity": entity_key, "mentions": count},
                    )
                )

        return gaps

    def _find_stalled_tasks(
        self,
        sessions: list[UniversalContextSchema],
        ages: list[float],
    ) -> list[KnowledgeGap]:
        """Find tasks that have been active for too long."""
        cfg = self._config

        # Collect active tasks with their oldest session age
        active_tasks: dict[str, tuple[Task, float]] = {}

        for session_idx, ucs in enumerate(sessions):
            age = ages[session_idx]
            for task in ucs.tasks:
                if task.status != TaskStatus.ACTIVE:
                    continue
                key = task.description.lower().strip()
                if key not in active_tasks or age > active_tasks[key][1]:
                    active_tasks[key] = (task, age)

        # Check if completed in any session
        completed_keys: set[str] = set()
        for ucs in sessions:
            for task in ucs.tasks:
                if task.status == TaskStatus.COMPLETED:
                    completed_keys.add(task.description.lower().strip())

        gaps: list[KnowledgeGap] = []
        for key, (task, oldest_age) in active_tasks.items():
            if key in completed_keys:
                continue
            if oldest_age >= cfg.stalled_task_age_days:
                gaps.append(
                    KnowledgeGap(
                        gap_type="stalled_task",
                        title=f"Stalled task: '{task.description[:60]}'",
                        detail=(
                            f"This task has been active for {oldest_age:.0f} days "
                            f"without completion (threshold: {cfg.stalled_task_age_days:.0f} days)."
                        ),
                        severity="high" if oldest_age >= cfg.stalled_task_age_days * 2 else "medium",
                        session_count=1,
                        metadata={
                            "task": task.description,
                            "age_days": oldest_age,
                            "priority": task.priority,
                        },
                    )
                )

        return gaps

    def _find_unclear_entities(
        self, sessions: list[UniversalContextSchema]
    ) -> list[KnowledgeGap]:
        """Find entities mentioned often but with low importance (poorly defined)."""
        cfg = self._config

        # Count mentions and track importance
        entity_data: dict[str, tuple[int, float, str]] = {}  # key -> (count, max_importance, name)

        for ucs in sessions:
            seen: set[str] = set()
            for entity in ucs.entities:
                key = entity.name.lower().strip()
                if key not in seen:
                    if key in entity_data:
                        count, max_imp, name = entity_data[key]
                        entity_data[key] = (
                            count + 1,
                            max(max_imp, entity.importance),
                            name,
                        )
                    else:
                        entity_data[key] = (1, entity.importance, entity.name)
                    seen.add(key)

        gaps: list[KnowledgeGap] = []
        for key, (count, max_imp, name) in entity_data.items():
            if (
                count >= cfg.min_entity_mentions
                and max_imp < cfg.low_importance_threshold
            ):
                gaps.append(
                    KnowledgeGap(
                        gap_type="unclear_entity",
                        title=f"Unclear entity: '{name}'",
                        detail=(
                            f"'{name}' appears in {count} sessions but has low "
                            f"importance ({max_imp:.2f}), suggesting it may need clarification."
                        ),
                        severity="low",
                        session_count=count,
                        metadata={
                            "entity": name,
                            "mentions": count,
                            "max_importance": max_imp,
                        },
                    )
                )

        return gaps

    def _count_unique_entities(
        self, sessions: list[UniversalContextSchema]
    ) -> int:
        seen: set[str] = set()
        for ucs in sessions:
            for e in ucs.entities:
                seen.add(e.name.lower().strip())
        return len(seen)

    def _count_unique_decisions(
        self, sessions: list[UniversalContextSchema]
    ) -> int:
        seen: set[str] = set()
        for ucs in sessions:
            for d in ucs.decisions:
                seen.add(d.description.lower().strip())
        return len(seen)

    def _count_unique_tasks(
        self, sessions: list[UniversalContextSchema]
    ) -> int:
        seen: set[str] = set()
        for ucs in sessions:
            for t in ucs.tasks:
                seen.add(t.description.lower().strip())
        return len(seen)
