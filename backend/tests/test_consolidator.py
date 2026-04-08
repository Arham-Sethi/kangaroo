"""Tests for MemoryConsolidator — merge multiple UCS sessions into one brain.

Tests cover:
    - Entity deduplication across sessions
    - Decision merging (unique, superseded handling)
    - Task merging (dedup by description, completed wins)
    - Preference merging (most recent wins, domains union)
    - Knowledge graph unification
    - Temporal decay integration
    - Empty and single-session cases
    - ConsolidationResult metadata
"""

import pytest

from app.core.brain.consolidator import ConsolidationResult, MemoryConsolidator
from app.core.brain.temporal import DecayConfig
from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    DetailLevel,
    Entity,
    EntityType,
    Preferences,
    SessionMeta,
    SourceLLM,
    Task,
    TaskStatus,
    TonePreference,
    UniversalContextSchema,
)


# -- Helpers -----------------------------------------------------------------


def _ucs(
    entities: list[Entity] | None = None,
    decisions: list[Decision] | None = None,
    tasks: list[Task] | None = None,
    preferences: Preferences | None = None,
    source_llm: SourceLLM = SourceLLM.OPENAI,
    message_count: int = 5,
) -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=source_llm,
            source_model="test",
            message_count=message_count,
            total_tokens=500,
        ),
        entities=tuple(entities or []),
        decisions=tuple(decisions or []),
        tasks=tuple(tasks or []),
        preferences=preferences or Preferences(),
    )


def _entity(name: str, etype: EntityType = EntityType.TECHNOLOGY, importance: float = 0.5) -> Entity:
    return Entity(name=name, type=etype, importance=importance)


def _decision(desc: str, status: DecisionStatus = DecisionStatus.ACTIVE) -> Decision:
    return Decision(description=desc, status=status)


def _task(desc: str, status: TaskStatus = TaskStatus.ACTIVE, priority: float = 0.5) -> Task:
    return Task(description=desc, status=status, priority=priority)


# -- Tests -------------------------------------------------------------------


class TestConsolidatorEmpty:
    def test_empty_input(self) -> None:
        c = MemoryConsolidator()
        result = c.consolidate([])
        assert result.session_count == 0
        assert result.entity_count == 0
        assert result.decision_count == 0

    def test_single_session_passthrough(self) -> None:
        ucs = _ucs(
            entities=[_entity("Python")],
            decisions=[_decision("Use Python")],
            tasks=[_task("Write tests")],
        )
        c = MemoryConsolidator()
        result = c.consolidate([ucs])
        assert result.session_count == 1
        assert result.entity_count == 1
        assert result.decision_count == 1
        assert result.task_count == 1


class TestConsolidatorEntities:
    def test_dedup_same_name(self) -> None:
        ucs_a = _ucs(entities=[_entity("Python", importance=0.8)])
        ucs_b = _ucs(entities=[_entity("Python", importance=0.6)])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.entity_count == 1
        assert result.entities_merged > 0
        # Should keep higher importance
        python = result.brain.entities[0]
        assert python.name == "Python"
        assert python.importance >= 0.6

    def test_different_entities_preserved(self) -> None:
        ucs_a = _ucs(entities=[_entity("Python")])
        ucs_b = _ucs(entities=[_entity("Rust")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.entity_count == 2

    def test_case_insensitive_dedup(self) -> None:
        ucs_a = _ucs(entities=[_entity("FastAPI")])
        ucs_b = _ucs(entities=[_entity("fastapi")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.entity_count == 1

    def test_merged_entity_has_session_count(self) -> None:
        ucs_a = _ucs(entities=[_entity("Python")])
        ucs_b = _ucs(entities=[_entity("Python")])
        ucs_c = _ucs(entities=[_entity("Python")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b, ucs_c])
        python = result.brain.entities[0]
        assert python.metadata.get("session_count", 0) >= 2

    def test_temporal_decay_reduces_old_entities(self) -> None:
        ucs_old = _ucs(entities=[_entity("OldTech", importance=0.9)])
        ucs_new = _ucs(entities=[_entity("NewTech", importance=0.5)])

        c = MemoryConsolidator(decay_config=DecayConfig(half_life_days=30))
        result = c.consolidate(
            [ucs_old, ucs_new],
            session_ages_days=[60, 0],  # old is 60 days, new is today
        )

        entities_by_name = {e.name: e for e in result.brain.entities}
        # Old entity should have decayed significantly
        assert entities_by_name["OldTech"].importance < 0.5
        # New entity should be untouched
        assert entities_by_name["NewTech"].importance == 0.5

    def test_entities_sorted_by_importance(self) -> None:
        ucs = _ucs(entities=[
            _entity("Low", importance=0.2),
            _entity("High", importance=0.9),
            _entity("Mid", importance=0.5),
        ])

        c = MemoryConsolidator()
        result = c.consolidate([ucs])
        names = [e.name for e in result.brain.entities]
        assert names == ["High", "Mid", "Low"]


class TestConsolidatorDecisions:
    def test_unique_decisions_preserved(self) -> None:
        ucs_a = _ucs(decisions=[_decision("Use REST")])
        ucs_b = _ucs(decisions=[_decision("Use PostgreSQL")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.decision_count == 2

    def test_duplicate_decisions_deduped(self) -> None:
        ucs_a = _ucs(decisions=[_decision("Use Python 3.12")])
        ucs_b = _ucs(decisions=[_decision("Use Python 3.12")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.decision_count == 1

    def test_active_wins_over_superseded(self) -> None:
        ucs_a = _ucs(decisions=[
            _decision("Use REST", status=DecisionStatus.SUPERSEDED),
        ])
        ucs_b = _ucs(decisions=[
            _decision("Use REST", status=DecisionStatus.ACTIVE),
        ])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.decision_count == 1
        assert result.brain.decisions[0].status == DecisionStatus.ACTIVE


class TestConsolidatorTasks:
    def test_unique_tasks_preserved(self) -> None:
        ucs_a = _ucs(tasks=[_task("Write tests")])
        ucs_b = _ucs(tasks=[_task("Deploy to staging")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.task_count == 2

    def test_duplicate_tasks_deduped(self) -> None:
        ucs_a = _ucs(tasks=[_task("Write tests")])
        ucs_b = _ucs(tasks=[_task("Write tests")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.task_count == 1

    def test_completed_wins(self) -> None:
        ucs_a = _ucs(tasks=[_task("Write tests", status=TaskStatus.ACTIVE)])
        ucs_b = _ucs(tasks=[_task("Write tests", status=TaskStatus.COMPLETED)])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.brain.tasks[0].status == TaskStatus.COMPLETED

    def test_higher_priority_wins(self) -> None:
        ucs_a = _ucs(tasks=[_task("Fix bug", priority=0.3)])
        ucs_b = _ucs(tasks=[_task("Fix bug", priority=0.9)])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.brain.tasks[0].priority == 0.9


class TestConsolidatorPreferences:
    def test_most_recent_tone_wins(self) -> None:
        ucs_old = _ucs(
            preferences=Preferences(tone=TonePreference.CASUAL),
        )
        ucs_new = _ucs(
            preferences=Preferences(tone=TonePreference.TECHNICAL),
        )

        c = MemoryConsolidator()
        result = c.consolidate([ucs_old, ucs_new], session_ages_days=[30, 0])
        assert result.brain.preferences.tone == TonePreference.TECHNICAL

    def test_domain_expertise_unioned(self) -> None:
        ucs_a = _ucs(
            preferences=Preferences(domain_expertise=("python", "ml")),
        )
        ucs_b = _ucs(
            preferences=Preferences(domain_expertise=("rust", "systems")),
        )

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        domains = set(result.brain.preferences.domain_expertise)
        assert domains == {"ml", "python", "rust", "systems"}


class TestConsolidatorKnowledgeGraph:
    def test_graph_has_nodes_for_entities(self) -> None:
        ucs = _ucs(entities=[_entity("Python"), _entity("FastAPI")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs])
        assert len(result.brain.knowledge_graph.nodes) == 2


class TestConsolidationResult:
    def test_result_metadata(self) -> None:
        ucs_a = _ucs(
            entities=[_entity("Python")],
            decisions=[_decision("Use Python")],
            tasks=[_task("Write code")],
        )
        ucs_b = _ucs(
            entities=[_entity("Rust")],
            decisions=[_decision("Use Rust")],
        )

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.session_count == 2
        assert result.entity_count == 2
        assert result.decision_count == 2
        assert result.task_count == 1

    def test_brain_message_count_summed(self) -> None:
        ucs_a = _ucs(message_count=10)
        ucs_b = _ucs(message_count=20)

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert result.brain.session_meta.message_count == 30

    def test_conflicts_detected(self) -> None:
        ucs_a = _ucs(decisions=[_decision("Use REST API")])
        ucs_b = _ucs(decisions=[_decision("Use GraphQL API")])

        c = MemoryConsolidator()
        result = c.consolidate([ucs_a, ucs_b])
        assert len(result.conflicts) >= 1
