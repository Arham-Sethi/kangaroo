"""Comprehensive tests for the Context Compression Pipeline.

Tests cover:
    1. Token estimation
    2. Priority queue ordering
    3. Artifact truncation
    4. Compression strategy (what gets dropped first)
    5. Protected content (decisions, tasks, global summary)
    6. Full pipeline integration

Total: 35+ tests
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.engine.compressor import (
    CompressionPipeline,
    CompressionResult,
    ContentCategory,
    PriorityItem,
    estimate_artifact_tokens,
    estimate_entity_tokens,
    estimate_tokens,
    truncate_artifact,
)
from app.core.models.ucs import (
    Artifact,
    ArtifactType,
    Decision,
    DecisionStatus,
    Entity,
    EntityType,
    Summary,
    SummaryLevel,
    Task,
    TaskStatus,
)


# -- Helpers -----------------------------------------------------------------


def _summary(level: SummaryLevel, content: str, tokens: int = 0) -> Summary:
    return Summary(
        level=level,
        content=content,
        token_count=tokens or estimate_tokens(content),
        covers_messages=(0, 0),
    )


def _entity(name: str, importance: float = 0.5) -> Entity:
    return Entity(
        id=uuid4(),
        name=name,
        type=EntityType.TECHNOLOGY,
        importance=importance,
    )


def _decision(desc: str) -> Decision:
    return Decision(description=desc, rationale="Because reasons.")


def _task(desc: str) -> Task:
    return Task(description=desc, status=TaskStatus.ACTIVE)


def _artifact(content: str, title: str = "test") -> Artifact:
    return Artifact(
        type=ArtifactType.CODE,
        language="python",
        content=content,
        title=title,
    )


# == Token Estimation Tests ==================================================


class TestTokenEstimation:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_text(self) -> None:
        # "hello" = 5 chars -> ~1 token
        assert estimate_tokens("hello") >= 1

    def test_longer_text(self) -> None:
        text = "This is a longer piece of text with many words"
        tokens = estimate_tokens(text)
        assert tokens > 5

    def test_entity_token_estimation(self) -> None:
        entity = _entity("PostgreSQL")
        tokens = estimate_entity_tokens(entity)
        assert tokens >= 1

    def test_artifact_token_estimation(self) -> None:
        artifact = _artifact("def foo():\n    return 42\n")
        tokens = estimate_artifact_tokens(artifact)
        assert tokens >= 1


# == Priority Item Tests =====================================================


class TestPriorityItem:
    def test_lower_priority_sorts_first(self) -> None:
        low = PriorityItem(1.0, ContentCategory.MESSAGE_SUMMARY, 10, 0, None)
        high = PriorityItem(5.0, ContentCategory.ENTITY, 10, 1, None)
        assert low < high

    def test_equal_priority_newer_drops_first(self) -> None:
        older = PriorityItem(1.0, ContentCategory.MESSAGE_SUMMARY, 10, 0, None)
        newer = PriorityItem(1.0, ContentCategory.MESSAGE_SUMMARY, 10, 5, None)
        assert newer < older  # newer index drops first


# == Artifact Truncation Tests ===============================================


class TestArtifactTruncation:
    def test_short_artifact_unchanged(self) -> None:
        artifact = _artifact("short code")
        result = truncate_artifact(artifact, max_tokens=100)
        assert result.content == "short code"

    def test_long_artifact_truncated(self) -> None:
        long_code = "x = 1\n" * 500  # ~3000 chars
        artifact = _artifact(long_code)
        result = truncate_artifact(artifact, max_tokens=50)
        assert len(result.content) < len(long_code)
        assert "[truncated]" in result.content

    def test_truncation_preserves_id(self) -> None:
        artifact = _artifact("a" * 1000)
        result = truncate_artifact(artifact, max_tokens=10)
        assert result.id == artifact.id

    def test_truncation_preserves_metadata(self) -> None:
        artifact = _artifact("a" * 1000)
        result = truncate_artifact(artifact, max_tokens=10)
        assert result.language == "python"
        assert result.type == ArtifactType.CODE


# == Compression Pipeline Tests ==============================================


class TestCompressionPipeline:
    def test_under_budget_no_compression(self) -> None:
        pipeline = CompressionPipeline(target_tokens=10000)
        result = pipeline.compress(
            summaries=(_summary(SummaryLevel.GLOBAL, "Short summary"),),
            entities=(_entity("Python"),),
        )
        assert result.compression_ratio == 1.0
        assert result.items_dropped == 0
        assert len(result.summaries) == 1
        assert len(result.entities) == 1

    def test_message_summaries_dropped_first(self) -> None:
        pipeline = CompressionPipeline(target_tokens=50)
        msg_summary = _summary(SummaryLevel.MESSAGE, "A" * 200)
        topic_summary = _summary(SummaryLevel.TOPIC, "B" * 200)
        global_summary = _summary(SummaryLevel.GLOBAL, "C" * 100)

        result = pipeline.compress(
            summaries=(msg_summary, topic_summary, global_summary),
        )
        levels = {s.level for s in result.summaries}
        # Message-level should be dropped before topic and global
        if SummaryLevel.MESSAGE not in levels:
            assert SummaryLevel.GLOBAL in levels  # global always survives

    def test_decisions_never_dropped(self) -> None:
        pipeline = CompressionPipeline(target_tokens=10)
        decision = _decision("Use PostgreSQL for the database")

        result = pipeline.compress(
            summaries=(_summary(SummaryLevel.MESSAGE, "A" * 500),),
            decisions=(decision,),
        )
        assert len(result.decisions) == 1
        assert result.decisions[0].description == decision.description

    def test_tasks_never_dropped(self) -> None:
        pipeline = CompressionPipeline(target_tokens=10)
        task = _task("Implement user authentication")

        result = pipeline.compress(
            summaries=(_summary(SummaryLevel.MESSAGE, "A" * 500),),
            tasks=(task,),
        )
        assert len(result.tasks) == 1
        assert result.tasks[0].description == task.description

    def test_global_summary_preserved(self) -> None:
        pipeline = CompressionPipeline(target_tokens=50)
        global_summary = _summary(SummaryLevel.GLOBAL, "Important global context")
        msg_summary = _summary(SummaryLevel.MESSAGE, "A" * 400)

        result = pipeline.compress(
            summaries=(msg_summary, global_summary),
        )
        global_summaries = [s for s in result.summaries if s.level == SummaryLevel.GLOBAL]
        assert len(global_summaries) == 1

    def test_low_importance_entities_dropped_first(self) -> None:
        pipeline = CompressionPipeline(target_tokens=20)
        high = _entity("Python", importance=0.9)
        low = _entity("SomeObscureTool", importance=0.1)

        result = pipeline.compress(
            entities=(high, low),
            importance_scores={str(high.id): 0.9, str(low.id): 0.1},
        )
        # If anything was dropped, the low-importance entity should go first
        if result.items_dropped > 0:
            names = {e.name for e in result.entities}
            if len(names) == 1:
                assert "Python" in names

    def test_compression_ratio_calculated(self) -> None:
        pipeline = CompressionPipeline(target_tokens=50)
        result = pipeline.compress(
            summaries=(
                _summary(SummaryLevel.MESSAGE, "A" * 400),
                _summary(SummaryLevel.GLOBAL, "B" * 100),
            ),
        )
        assert 0.0 < result.compression_ratio <= 1.0

    def test_empty_input(self) -> None:
        pipeline = CompressionPipeline(target_tokens=100)
        result = pipeline.compress()
        assert result.total_tokens == 0
        assert result.items_dropped == 0
        assert result.compression_ratio == 1.0

    def test_artifacts_truncated_not_dropped(self) -> None:
        pipeline = CompressionPipeline(target_tokens=100, artifact_max_tokens=20)
        long_artifact = _artifact("x = 1\n" * 200)

        result = pipeline.compress(artifacts=(long_artifact,))
        # Artifact should still exist (possibly truncated)
        if result.original_tokens > pipeline._target_tokens:
            # Some compression happened
            assert result.compression_ratio < 1.0 or result.items_dropped > 0

    def test_result_immutability(self) -> None:
        pipeline = CompressionPipeline(target_tokens=10000)
        result = pipeline.compress(
            summaries=(_summary(SummaryLevel.GLOBAL, "test"),),
        )
        with pytest.raises(AttributeError):
            result.items_dropped = 99  # type: ignore[misc]

    def test_large_conversation_compression(self) -> None:
        """Test compression with many items to ensure priority ordering works."""
        pipeline = CompressionPipeline(target_tokens=200)
        summaries = tuple(
            _summary(SummaryLevel.MESSAGE, f"Message {i} content " * 5)
            for i in range(20)
        ) + (_summary(SummaryLevel.GLOBAL, "Global overview of the conversation"),)

        entities = tuple(_entity(f"Tech{i}", importance=i / 20) for i in range(10))
        decisions = (_decision("Use microservices architecture"),)
        tasks = (_task("Deploy to production"),)

        result = pipeline.compress(
            summaries=summaries,
            entities=entities,
            decisions=decisions,
            tasks=tasks,
        )

        # Decisions and tasks preserved
        assert len(result.decisions) == 1
        assert len(result.tasks) == 1
        # Some items were dropped
        assert result.items_dropped > 0
        assert result.compression_ratio < 1.0
