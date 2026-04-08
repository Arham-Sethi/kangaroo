"""Tests for GapDetector — find knowledge gaps across sessions.

Tests cover:
    - Empty sessions
    - Undecided topic detection
    - Stalled task detection
    - Unclear entity detection
    - Severity classification
    - Config customization
    - Stats in report
    - Gap sorting by severity
"""

import pytest

from app.core.brain.gaps import (
    GapConfig,
    GapDetector,
    GapReport,
    KnowledgeGap,
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
) -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="test",
            message_count=5,
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


class TestGapDetectorEmpty:
    def test_empty_sessions(self) -> None:
        detector = GapDetector()
        report = detector.detect([])
        assert report.sessions_analyzed == 0
        assert report.gaps == ()

    def test_single_session_no_gaps(self) -> None:
        detector = GapDetector()
        report = detector.detect([_ucs(entities=[_entity("Python")])])
        # Single session can't have undecided topics (needs >= 2 mentions)
        undecided = [g for g in report.gaps if g.gap_type == "undecided_topic"]
        assert len(undecided) == 0


class TestUndecidedTopics:
    def test_entity_mentioned_twice_without_decision(self) -> None:
        detector = GapDetector(config=GapConfig(min_mentions_for_undecided=2))
        sessions = [
            _ucs(entities=[_entity("Redis")]),
            _ucs(entities=[_entity("Redis")]),
        ]
        report = detector.detect(sessions)
        undecided = [g for g in report.gaps if g.gap_type == "undecided_topic"]
        assert len(undecided) >= 1
        assert any("redis" in g.title.lower() for g in undecided)

    def test_no_gap_if_decision_exists(self) -> None:
        detector = GapDetector(config=GapConfig(min_mentions_for_undecided=2))
        sessions = [
            _ucs(entities=[_entity("Redis")], decisions=[_decision("Use Redis for caching")]),
            _ucs(entities=[_entity("Redis")]),
        ]
        report = detector.detect(sessions)
        undecided = [g for g in report.gaps if g.gap_type == "undecided_topic"]
        # Redis has a decision, so no gap
        redis_gaps = [g for g in undecided if "redis" in g.title.lower()]
        assert len(redis_gaps) == 0

    def test_high_severity_for_many_mentions(self) -> None:
        detector = GapDetector(config=GapConfig(min_mentions_for_undecided=2))
        sessions = [
            _ucs(entities=[_entity("Kafka")]),
            _ucs(entities=[_entity("Kafka")]),
            _ucs(entities=[_entity("Kafka")]),
        ]
        report = detector.detect(sessions)
        undecided = [g for g in report.gaps if g.gap_type == "undecided_topic"]
        kafka_gaps = [g for g in undecided if "kafka" in g.title.lower()]
        assert len(kafka_gaps) >= 1
        assert kafka_gaps[0].severity == "high"

    def test_below_threshold_not_flagged(self) -> None:
        detector = GapDetector(config=GapConfig(min_mentions_for_undecided=3))
        sessions = [
            _ucs(entities=[_entity("Kubernetes")]),
            _ucs(entities=[_entity("Kubernetes")]),
        ]
        report = detector.detect(sessions)
        undecided = [g for g in report.gaps if g.gap_type == "undecided_topic"]
        k8s_gaps = [g for g in undecided if "kubernetes" in g.title.lower()]
        assert len(k8s_gaps) == 0


class TestStalledTasks:
    def test_old_active_task_flagged(self) -> None:
        detector = GapDetector(config=GapConfig(stalled_task_age_days=14))
        sessions = [
            _ucs(tasks=[_task("Migrate database", status=TaskStatus.ACTIVE)]),
        ]
        report = detector.detect(sessions, session_ages_days=[30])
        stalled = [g for g in report.gaps if g.gap_type == "stalled_task"]
        assert len(stalled) >= 1

    def test_recent_task_not_flagged(self) -> None:
        detector = GapDetector(config=GapConfig(stalled_task_age_days=14))
        sessions = [
            _ucs(tasks=[_task("Write tests", status=TaskStatus.ACTIVE)]),
        ]
        report = detector.detect(sessions, session_ages_days=[5])
        stalled = [g for g in report.gaps if g.gap_type == "stalled_task"]
        assert len(stalled) == 0

    def test_completed_task_not_stalled(self) -> None:
        detector = GapDetector(config=GapConfig(stalled_task_age_days=14))
        sessions = [
            _ucs(tasks=[_task("Old task", status=TaskStatus.ACTIVE)]),
            _ucs(tasks=[_task("Old task", status=TaskStatus.COMPLETED)]),
        ]
        report = detector.detect(sessions, session_ages_days=[30, 0])
        stalled = [g for g in report.gaps if g.gap_type == "stalled_task"]
        old_task_gaps = [g for g in stalled if "old task" in g.title.lower()]
        assert len(old_task_gaps) == 0

    def test_very_old_task_high_severity(self) -> None:
        detector = GapDetector(config=GapConfig(stalled_task_age_days=14))
        sessions = [
            _ucs(tasks=[_task("Ancient task", status=TaskStatus.ACTIVE)]),
        ]
        report = detector.detect(sessions, session_ages_days=[60])
        stalled = [g for g in report.gaps if g.gap_type == "stalled_task"]
        assert len(stalled) >= 1
        assert stalled[0].severity == "high"


class TestUnclearEntities:
    def test_low_importance_frequent_entity(self) -> None:
        detector = GapDetector(config=GapConfig(
            min_entity_mentions=3,
            low_importance_threshold=0.3,
        ))
        sessions = [
            _ucs(entities=[_entity("Vague Thing", importance=0.1)]),
            _ucs(entities=[_entity("Vague Thing", importance=0.2)]),
            _ucs(entities=[_entity("Vague Thing", importance=0.15)]),
        ]
        report = detector.detect(sessions)
        unclear = [g for g in report.gaps if g.gap_type == "unclear_entity"]
        assert len(unclear) >= 1

    def test_high_importance_not_flagged(self) -> None:
        detector = GapDetector(config=GapConfig(
            min_entity_mentions=2,
            low_importance_threshold=0.3,
        ))
        sessions = [
            _ucs(entities=[_entity("Clear Thing", importance=0.8)]),
            _ucs(entities=[_entity("Clear Thing", importance=0.9)]),
            _ucs(entities=[_entity("Clear Thing", importance=0.7)]),
        ]
        report = detector.detect(sessions)
        unclear = [g for g in report.gaps if g.gap_type == "unclear_entity"]
        clear_gaps = [g for g in unclear if "clear thing" in g.title.lower()]
        assert len(clear_gaps) == 0

    def test_few_mentions_not_flagged(self) -> None:
        detector = GapDetector(config=GapConfig(min_entity_mentions=5))
        sessions = [
            _ucs(entities=[_entity("Rare", importance=0.1)]),
            _ucs(entities=[_entity("Rare", importance=0.1)]),
        ]
        report = detector.detect(sessions)
        unclear = [g for g in report.gaps if g.gap_type == "unclear_entity"]
        assert len(unclear) == 0


class TestGapReport:
    def test_report_stats(self) -> None:
        detector = GapDetector()
        sessions = [
            _ucs(
                entities=[_entity("Python"), _entity("Rust")],
                decisions=[_decision("Use Python")],
                tasks=[_task("Write tests")],
            ),
            _ucs(
                entities=[_entity("Python")],
                decisions=[_decision("Deploy on AWS")],
            ),
        ]
        report = detector.detect(sessions)
        assert report.sessions_analyzed == 2
        assert report.total_entities == 2  # Python + Rust (deduped)
        assert report.total_decisions == 2
        assert report.total_tasks == 1

    def test_gaps_sorted_by_severity(self) -> None:
        detector = GapDetector(config=GapConfig(
            min_mentions_for_undecided=2,
            stalled_task_age_days=14,
            min_entity_mentions=2,
            low_importance_threshold=0.3,
        ))
        sessions = [
            _ucs(
                entities=[_entity("Topic", importance=0.1)],
                tasks=[_task("Old task", status=TaskStatus.ACTIVE)],
            ),
            _ucs(entities=[_entity("Topic", importance=0.1)]),
            _ucs(entities=[_entity("Topic", importance=0.1)]),
        ]
        report = detector.detect(sessions, session_ages_days=[30, 10, 0])

        if len(report.gaps) >= 2:
            severity_order = {"high": 0, "medium": 1, "low": 2}
            severities = [severity_order.get(g.severity, 99) for g in report.gaps]
            assert severities == sorted(severities)


class TestGapConfig:
    def test_default_config(self) -> None:
        cfg = GapConfig()
        assert cfg.min_mentions_for_undecided == 2
        assert cfg.stalled_task_age_days == 14.0
        assert cfg.low_importance_threshold == 0.3
        assert cfg.min_entity_mentions == 3

    def test_custom_config(self) -> None:
        detector = GapDetector(config=GapConfig(
            min_mentions_for_undecided=5,
            stalled_task_age_days=7,
        ))
        assert detector.config.min_mentions_for_undecided == 5
        assert detector.config.stalled_task_age_days == 7.0
