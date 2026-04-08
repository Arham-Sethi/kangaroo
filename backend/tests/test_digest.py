"""Tests for DigestGenerator — summarize recent sessions.

Tests cover:
    - Empty sessions
    - Single session digest
    - Entity collection and dedup
    - Decision collection (active only)
    - Task collection (completed status)
    - Config thresholds (min_entity_importance)
    - Digest stats (session_count, total_messages)
    - Sorting by importance
"""

import pytest

from app.core.brain.digest import (
    Digest,
    DigestConfig,
    DigestEntry,
    DigestGenerator,
)
from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    Entity,
    EntityType,
    Preferences,
    SessionMeta,
    SourceLLM,
    Task,
    TaskStatus,
    UniversalContextSchema,
)


# -- Helpers -----------------------------------------------------------------


def _ucs(
    entities: list[Entity] | None = None,
    decisions: list[Decision] | None = None,
    tasks: list[Task] | None = None,
    message_count: int = 10,
) -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="test",
            message_count=message_count,
            total_tokens=500,
        ),
        entities=tuple(entities or []),
        decisions=tuple(decisions or []),
        tasks=tuple(tasks or []),
    )


def _entity(name: str, importance: float = 0.5) -> Entity:
    return Entity(name=name, type=EntityType.TECHNOLOGY, importance=importance)


def _decision(desc: str, status: DecisionStatus = DecisionStatus.ACTIVE) -> Decision:
    return Decision(description=desc, status=status)


def _task(desc: str, status: TaskStatus = TaskStatus.ACTIVE, priority: float = 0.5) -> Task:
    return Task(description=desc, status=status, priority=priority)


# -- Tests -------------------------------------------------------------------


class TestDigestEmpty:
    def test_empty_sessions(self) -> None:
        gen = DigestGenerator()
        digest = gen.generate([])
        assert digest.session_count == 0
        assert digest.entries == ()
        assert digest.new_entities == 0
        assert digest.decisions_made == 0
        assert digest.tasks_completed == 0

    def test_empty_period_label(self) -> None:
        gen = DigestGenerator()
        digest = gen.generate([], period_label="Yesterday")
        assert digest.period_label == "Yesterday"


class TestDigestEntities:
    def test_collects_entities(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(entities=[_entity("Python", 0.8), _entity("FastAPI", 0.6)])]
        digest = gen.generate(sessions)
        assert digest.new_entities == 2
        entity_titles = [e.title for e in digest.entries if e.category == "entity"]
        assert "Python" in entity_titles
        assert "FastAPI" in entity_titles

    def test_deduplicates_entities(self) -> None:
        gen = DigestGenerator()
        sessions = [
            _ucs(entities=[_entity("Python", 0.7)]),
            _ucs(entities=[_entity("Python", 0.9)]),
        ]
        digest = gen.generate(sessions)
        assert digest.new_entities == 1
        entity_entries = [e for e in digest.entries if e.category == "entity"]
        assert len(entity_entries) == 1

    def test_filters_low_importance(self) -> None:
        gen = DigestGenerator(config=DigestConfig(min_entity_importance=0.5))
        sessions = [_ucs(entities=[
            _entity("Important", 0.8),
            _entity("Weak", 0.2),
        ])]
        digest = gen.generate(sessions)
        entity_titles = [e.title for e in digest.entries if e.category == "entity"]
        assert "Important" in entity_titles
        assert "Weak" not in entity_titles

    def test_max_entities_limit(self) -> None:
        gen = DigestGenerator(config=DigestConfig(max_entities=2))
        entities = [_entity(f"Entity{i}", importance=0.5 + i * 0.01) for i in range(10)]
        sessions = [_ucs(entities=entities)]
        digest = gen.generate(sessions)
        entity_entries = [e for e in digest.entries if e.category == "entity"]
        assert len(entity_entries) <= 2


class TestDigestDecisions:
    def test_collects_active_decisions(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(decisions=[
            _decision("Use REST API"),
            _decision("Use PostgreSQL"),
        ])]
        digest = gen.generate(sessions)
        assert digest.decisions_made == 2

    def test_ignores_superseded_decisions(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(decisions=[
            _decision("Use REST", status=DecisionStatus.ACTIVE),
            _decision("Use GraphQL", status=DecisionStatus.SUPERSEDED),
        ])]
        digest = gen.generate(sessions)
        assert digest.decisions_made == 1

    def test_deduplicates_decisions(self) -> None:
        gen = DigestGenerator()
        sessions = [
            _ucs(decisions=[_decision("Use Python")]),
            _ucs(decisions=[_decision("Use Python")]),
        ]
        digest = gen.generate(sessions)
        decision_entries = [e for e in digest.entries if e.category == "decision"]
        assert len(decision_entries) == 1


class TestDigestTasks:
    def test_collects_tasks(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(tasks=[_task("Write tests"), _task("Deploy")])]
        digest = gen.generate(sessions)
        task_entries = [e for e in digest.entries if e.category == "task"]
        assert len(task_entries) == 2

    def test_completed_tasks_counted(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(tasks=[
            _task("Done task", status=TaskStatus.COMPLETED),
            _task("Active task", status=TaskStatus.ACTIVE),
        ])]
        digest = gen.generate(sessions)
        assert digest.tasks_completed == 1

    def test_completed_wins_dedup(self) -> None:
        gen = DigestGenerator()
        sessions = [
            _ucs(tasks=[_task("Write tests", status=TaskStatus.ACTIVE)]),
            _ucs(tasks=[_task("Write tests", status=TaskStatus.COMPLETED)]),
        ]
        digest = gen.generate(sessions)
        task_entries = [e for e in digest.entries if e.category == "task"]
        assert len(task_entries) == 1


class TestDigestStats:
    def test_session_count(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(), _ucs(), _ucs()]
        digest = gen.generate(sessions)
        assert digest.session_count == 3

    def test_total_messages(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(message_count=10), _ucs(message_count=20)]
        digest = gen.generate(sessions)
        assert digest.total_messages == 30

    def test_entries_sorted_by_importance(self) -> None:
        gen = DigestGenerator()
        sessions = [_ucs(
            entities=[_entity("Low", 0.3), _entity("High", 0.9)],
            decisions=[_decision("Medium decision")],
        )]
        digest = gen.generate(sessions)
        importances = [e.importance for e in digest.entries]
        assert importances == sorted(importances, reverse=True)

    def test_period_label(self) -> None:
        gen = DigestGenerator()
        digest = gen.generate([_ucs()], period_label="Last 7 days")
        assert digest.period_label == "Last 7 days"
