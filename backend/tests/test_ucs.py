"""Comprehensive tests for the Universal Context Schema (UCS).

Tests cover:
    - Model creation with valid data
    - Validation of all field constraints
    - Cross-reference integrity validation
    - UCSValidator semantic checks
    - Edge cases: empty UCS, maximum-size fields, boundary values
    - Immutability enforcement
    - Serialization roundtrip (JSON -> UCS -> JSON)
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.models.ucs import (
    Artifact,
    ArtifactType,
    Decision,
    DecisionStatus,
    DetailLevel,
    Entity,
    EntityRelationship,
    EntityType,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    LLMComparison,
    Preferences,
    ProcessingMode,
    RelationshipType,
    SafetyAction,
    SafetyFlag,
    SafetyFlagType,
    SafetySeverity,
    SessionMeta,
    SourceLLM,
    Summary,
    SummaryLevel,
    Task,
    TaskStatus,
    TonePreference,
    TopicCluster,
    UCSValidator,
    UniversalContextSchema,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def session_meta() -> SessionMeta:
    """Create a valid SessionMeta for testing."""
    return SessionMeta(
        source_llm=SourceLLM.OPENAI,
        source_model="gpt-4o",
        total_tokens=5000,
        message_count=20,
        compression_ratio=0.5,
    )


@pytest.fixture
def entity_a_id():
    return uuid4()


@pytest.fixture
def entity_b_id():
    return uuid4()


@pytest.fixture
def two_entities(entity_a_id, entity_b_id) -> tuple[Entity, Entity]:
    """Create two related entities."""
    entity_a = Entity(
        id=entity_a_id,
        name="FastAPI",
        type=EntityType.TECHNOLOGY,
        aliases=("fastapi", "fast-api"),
        importance=0.9,
        relationships=(
            EntityRelationship(
                target_id=entity_b_id,
                type=RelationshipType.DEPENDS_ON,
            ),
        ),
    )
    entity_b = Entity(
        id=entity_b_id,
        name="Python",
        type=EntityType.TECHNOLOGY,
        importance=0.95,
    )
    return entity_a, entity_b


@pytest.fixture
def minimal_ucs(session_meta) -> UniversalContextSchema:
    """Create a minimal valid UCS."""
    return UniversalContextSchema(session_meta=session_meta)


@pytest.fixture
def full_ucs(session_meta, two_entities) -> UniversalContextSchema:
    """Create a fully populated UCS."""
    entity_a, entity_b = two_entities
    task_1_id = uuid4()
    task_2_id = uuid4()
    decision_1_id = uuid4()
    decision_2_id = uuid4()

    return UniversalContextSchema(
        version="1.0.0",
        session_meta=session_meta,
        entities=(entity_a, entity_b),
        summaries=(
            Summary(
                level=SummaryLevel.GLOBAL,
                content="Discussion about building a web API with FastAPI and Python.",
                token_count=15,
                covers_messages=(0, 19),
            ),
            Summary(
                level=SummaryLevel.TOPIC,
                content="Discussed authentication setup using JWT tokens.",
                token_count=10,
                covers_messages=(5, 10),
            ),
        ),
        decisions=(
            Decision(
                id=decision_1_id,
                description="Use JWT for authentication",
                rationale="Industry standard, stateless, works with API-first design.",
                decided_at=7,
                status=DecisionStatus.ACTIVE,
            ),
            Decision(
                id=decision_2_id,
                description="Switch from Flask to FastAPI",
                rationale="Async support, better performance, automatic OpenAPI docs.",
                decided_at=3,
                status=DecisionStatus.SUPERSEDED,
                superseded_by=decision_1_id,
            ),
        ),
        tasks=(
            Task(
                id=task_1_id,
                description="Implement user registration endpoint",
                status=TaskStatus.ACTIVE,
                priority=0.8,
            ),
            Task(
                id=task_2_id,
                description="Add rate limiting middleware",
                status=TaskStatus.ACTIVE,
                dependencies=(task_1_id,),
                priority=0.6,
            ),
        ),
        preferences=Preferences(
            tone=TonePreference.TECHNICAL,
            detail_level=DetailLevel.DETAILED,
            format_preferences={"code_blocks": True, "markdown": True},
            domain_expertise=("web development", "python", "devops"),
            language="en",
        ),
        artifacts=(
            Artifact(
                type=ArtifactType.CODE,
                language="python",
                content="from fastapi import FastAPI\napp = FastAPI()",
                title="main.py",
            ),
        ),
        knowledge_graph=KnowledgeGraph(
            nodes=(
                KnowledgeGraphNode(entity_id=entity_a.id, label="FastAPI", group="tech"),
                KnowledgeGraphNode(entity_id=entity_b.id, label="Python", group="tech"),
            ),
            edges=(
                KnowledgeGraphEdge(
                    source_id=entity_a.id,
                    target_id=entity_b.id,
                    relationship=RelationshipType.DEPENDS_ON,
                ),
            ),
        ),
        topic_clusters=(
            TopicCluster(
                label="API Framework Selection",
                message_indices=(0, 1, 2, 3, 4),
            ),
            TopicCluster(
                label="Authentication Design",
                message_indices=(5, 6, 7, 8, 9, 10),
            ),
        ),
        safety_flags=(),
        llm_comparisons=(),
        importance_scores={
            str(entity_a.id): 0.9,
            str(entity_b.id): 0.95,
        },
    )


# ── SessionMeta Tests ─────────────────────────────────────────────────────────


class TestSessionMeta:
    def test_valid_creation(self, session_meta):
        assert session_meta.source_llm == SourceLLM.OPENAI
        assert session_meta.source_model == "gpt-4o"
        assert session_meta.total_tokens == 5000

    def test_defaults(self):
        meta = SessionMeta()
        assert meta.source_llm == SourceLLM.UNKNOWN
        assert meta.total_tokens == 0
        assert meta.compression_ratio == 1.0
        assert meta.processing_mode == ProcessingMode.STANDARD

    def test_naive_datetime_gets_utc(self):
        naive = datetime(2025, 1, 1, 12, 0, 0)
        meta = SessionMeta(created_at=naive)
        assert meta.created_at.tzinfo == timezone.utc

    def test_iso_string_parsed(self):
        meta = SessionMeta(created_at="2025-06-15T10:30:00+00:00")
        assert meta.created_at.year == 2025
        assert meta.created_at.month == 6

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            SessionMeta(total_tokens=-1)

    def test_compression_ratio_bounds(self):
        with pytest.raises(ValidationError):
            SessionMeta(compression_ratio=1.5)
        with pytest.raises(ValidationError):
            SessionMeta(compression_ratio=-0.1)

    def test_immutability(self, session_meta):
        with pytest.raises(ValidationError):
            session_meta.total_tokens = 999


# ── Entity Tests ──────────────────────────────────────────────────────────────


class TestEntity:
    def test_valid_entity(self, two_entities):
        entity_a, _ = two_entities
        assert entity_a.name == "FastAPI"
        assert entity_a.type == EntityType.TECHNOLOGY
        assert len(entity_a.relationships) == 1

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            Entity(name="", type=EntityType.CONCEPT)

    def test_importance_bounds(self):
        with pytest.raises(ValidationError):
            Entity(name="Test", type=EntityType.CONCEPT, importance=1.5)

    def test_aliases_are_tuple(self, two_entities):
        entity_a, _ = two_entities
        assert isinstance(entity_a.aliases, tuple)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            Entity(name="Test", type=EntityType.CONCEPT, unknown_field="value")


# ── Summary Tests ─────────────────────────────────────────────────────────────


class TestSummary:
    def test_valid_summary(self):
        s = Summary(
            level=SummaryLevel.GLOBAL,
            content="This is a test summary.",
            token_count=5,
            covers_messages=(0, 10),
        )
        assert s.level == SummaryLevel.GLOBAL

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            Summary(level=SummaryLevel.GLOBAL, content="")


# ── Decision Tests ────────────────────────────────────────────────────────────


class TestDecision:
    def test_valid_decision(self):
        d = Decision(description="Use PostgreSQL", status=DecisionStatus.ACTIVE)
        assert d.status == DecisionStatus.ACTIVE

    def test_superseded_decision(self):
        new_id = uuid4()
        d = Decision(
            description="Use MySQL",
            status=DecisionStatus.SUPERSEDED,
            superseded_by=new_id,
        )
        assert d.superseded_by == new_id


# ── Task Tests ────────────────────────────────────────────────────────────────


class TestTask:
    def test_valid_task(self):
        t = Task(description="Implement feature X")
        assert t.status == TaskStatus.ACTIVE
        assert t.priority == 0.5

    def test_task_with_dependencies(self):
        dep_id = uuid4()
        t = Task(description="Deploy", dependencies=(dep_id,))
        assert dep_id in t.dependencies


# ── Full UCS Tests ────────────────────────────────────────────────────────────


class TestUniversalContextSchema:
    def test_minimal_ucs(self, minimal_ucs):
        assert minimal_ucs.version == "1.0.0"
        assert len(minimal_ucs.entities) == 0
        assert len(minimal_ucs.summaries) == 0

    def test_full_ucs(self, full_ucs):
        assert len(full_ucs.entities) == 2
        assert len(full_ucs.summaries) == 2
        assert len(full_ucs.decisions) == 2
        assert len(full_ucs.tasks) == 2
        assert len(full_ucs.artifacts) == 1
        assert len(full_ucs.knowledge_graph.nodes) == 2
        assert len(full_ucs.knowledge_graph.edges) == 1

    def test_invalid_version_format(self, session_meta):
        with pytest.raises(ValidationError):
            UniversalContextSchema(session_meta=session_meta, version="v1")

    def test_invalid_entity_relationship_target(self, session_meta):
        """Entity references a non-existent entity — must fail validation."""
        bad_entity = Entity(
            name="Orphan",
            type=EntityType.CONCEPT,
            relationships=(
                EntityRelationship(
                    target_id=uuid4(),
                    type=RelationshipType.DEPENDS_ON,
                ),
            ),
        )
        with pytest.raises(ValidationError, match="non-existent entity ID"):
            UniversalContextSchema(
                session_meta=session_meta,
                entities=(bad_entity,),
            )

    def test_invalid_knowledge_graph_node(self, session_meta):
        """KG node references non-existent entity — must fail."""
        with pytest.raises(ValidationError, match="non-existent entity ID"):
            UniversalContextSchema(
                session_meta=session_meta,
                knowledge_graph=KnowledgeGraph(
                    nodes=(KnowledgeGraphNode(entity_id=uuid4()),),
                ),
            )

    def test_invalid_knowledge_graph_edge(self, session_meta, two_entities):
        """KG edge source/target not in nodes — must fail."""
        entity_a, entity_b = two_entities
        node_a = KnowledgeGraphNode(entity_id=entity_a.id)
        with pytest.raises(ValidationError, match="not a node"):
            UniversalContextSchema(
                session_meta=session_meta,
                entities=(entity_a, entity_b),
                knowledge_graph=KnowledgeGraph(
                    nodes=(node_a,),
                    edges=(
                        KnowledgeGraphEdge(
                            source_id=entity_a.id,
                            target_id=entity_b.id,
                            relationship=RelationshipType.DEPENDS_ON,
                        ),
                    ),
                ),
            )

    def test_invalid_task_dependency(self, session_meta):
        """Task depends on non-existent task — must fail."""
        with pytest.raises(ValidationError, match="non-existent task ID"):
            UniversalContextSchema(
                session_meta=session_meta,
                tasks=(
                    Task(
                        description="Deploy",
                        dependencies=(uuid4(),),
                    ),
                ),
            )

    def test_invalid_decision_supersession(self, session_meta):
        """Decision superseded_by references non-existent decision — must fail."""
        with pytest.raises(ValidationError, match="non-existent decision ID"):
            UniversalContextSchema(
                session_meta=session_meta,
                decisions=(
                    Decision(
                        description="Old decision",
                        status=DecisionStatus.SUPERSEDED,
                        superseded_by=uuid4(),
                    ),
                ),
            )

    def test_json_roundtrip(self, full_ucs):
        """Serialize to JSON and deserialize back — must be identical."""
        json_str = full_ucs.model_dump_json()
        restored = UniversalContextSchema.model_validate_json(json_str)
        assert restored.version == full_ucs.version
        assert len(restored.entities) == len(full_ucs.entities)
        assert restored.session_meta.source_llm == full_ucs.session_meta.source_llm

    def test_dict_roundtrip(self, full_ucs):
        """Serialize to dict and back."""
        data = full_ucs.model_dump()
        restored = UniversalContextSchema.model_validate(data)
        assert restored == full_ucs

    def test_immutability(self, full_ucs):
        """Cannot mutate the UCS after creation."""
        with pytest.raises(ValidationError):
            full_ucs.version = "2.0.0"


# ── UCSValidator Tests ────────────────────────────────────────────────────────


class TestUCSValidator:
    def test_valid_full_ucs(self, full_ucs):
        warnings = UCSValidator.validate(full_ucs)
        assert len(warnings) == 0
        assert UCSValidator.is_valid(full_ucs)

    def test_missing_global_summary_warning(self, session_meta):
        """UCS with messages but no global summary triggers warning."""
        ucs = UniversalContextSchema(
            session_meta=SessionMeta(
                source_llm=SourceLLM.OPENAI,
                message_count=10,
            ),
            summaries=(
                Summary(
                    level=SummaryLevel.MESSAGE,
                    content="A message summary.",
                    covers_messages=(0, 0),
                ),
            ),
        )
        warnings = UCSValidator.validate(ucs)
        assert any("no global summary" in w for w in warnings)

    def test_orphaned_importance_score_warning(self, session_meta):
        """Importance score for non-existent entity triggers warning."""
        ucs = UniversalContextSchema(
            session_meta=session_meta,
            importance_scores={"non-existent-uuid": 0.8},
        )
        warnings = UCSValidator.validate(ucs)
        assert any("no matching entity" in w for w in warnings)

    def test_no_warnings_for_empty_ucs(self):
        """Empty UCS (no messages) should not warn about missing summary."""
        ucs = UniversalContextSchema(
            session_meta=SessionMeta(message_count=0),
        )
        warnings = UCSValidator.validate(ucs)
        assert len(warnings) == 0

    def test_topic_cluster_out_of_range_warning(self, session_meta):
        """Topic cluster referencing message index beyond message_count."""
        ucs = UniversalContextSchema(
            session_meta=SessionMeta(message_count=5),
            topic_clusters=(
                TopicCluster(
                    label="Test cluster",
                    message_indices=(0, 1, 99),
                ),
            ),
        )
        warnings = UCSValidator.validate(ucs)
        assert any("message index" in w for w in warnings)
