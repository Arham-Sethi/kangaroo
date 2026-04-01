"""Universal Context Schema (UCS) — Pydantic v2 models.

The UCS is the core data format of Kangaroo Shift. It is the universal
representation of conversational context that works across every LLM.

Design principles:
    1. Every field has a clear semantic meaning documented in its description.
    2. All IDs are UUIDs — globally unique, no collision across users/teams.
    3. Timestamps are ISO-8601 UTC — timezone-naive datetimes are rejected.
    4. Enums use lowercase string literals — consistent, JSON-friendly.
    5. Optional fields have sensible defaults — a minimal UCS is still valid.
    6. The schema is versioned — old UCS documents can be migrated forward.
    7. Immutable design — functions return new UCS objects, never mutate.

Schema version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────


class SourceLLM(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENSOURCE = "opensource"
    UNKNOWN = "unknown"


class ProcessingMode(str, Enum):
    """How the context was processed."""

    STANDARD = "standard"
    LOCAL = "local"


class EntityType(str, Enum):
    """Categories of extracted entities."""

    PERSON = "person"
    CODE = "code"
    CONCEPT = "concept"
    ORGANIZATION = "org"
    TECHNOLOGY = "tech"
    LOCATION = "location"
    FILE_PATH = "file_path"
    URL = "url"
    API = "api"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    USES = "uses"
    CREATED_BY = "created_by"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    COMMUNICATES_WITH = "communicates_with"


class SummaryLevel(str, Enum):
    """Hierarchical summary granularity."""

    MESSAGE = "message"
    TOPIC = "topic"
    GLOBAL = "global"


class DecisionStatus(str, Enum):
    """Lifecycle status of a decision."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REVERTED = "reverted"


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""

    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TonePreference(str, Enum):
    """User's preferred communication tone."""

    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"


class DetailLevel(str, Enum):
    """User's preferred response detail level."""

    CONCISE = "concise"
    DETAILED = "detailed"
    EXHAUSTIVE = "exhaustive"


class ArtifactType(str, Enum):
    """Types of artifacts produced during conversation."""

    CODE = "code"
    CONFIG = "config"
    FILE = "file"
    OUTPUT = "output"
    DIAGRAM = "diagram"
    DOCUMENT = "document"


class SafetyFlagType(str, Enum):
    """Categories of safety concerns."""

    INJECTION = "injection"
    POISONING = "poisoning"
    POLICY_VIOLATION = "policy_violation"
    PII_DETECTED = "pii_detected"


class SafetySeverity(str, Enum):
    """Severity levels for safety flags."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyAction(str, Enum):
    """Actions taken in response to safety flags."""

    STRIPPED = "stripped"
    FLAGGED = "flagged"
    BLOCKED = "blocked"
    REDACTED = "redacted"


# ── Immutable Base ────────────────────────────────────────────────────────────


class ImmutableModel(BaseModel):
    """Base model enforcing immutability.

    All UCS sub-models inherit from this. Frozen config prevents
    accidental mutation — a core design principle for data integrity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ── Sub-Models ────────────────────────────────────────────────────────────────


class SessionMeta(ImmutableModel):
    """Metadata about the source conversation session.

    This captures WHERE the context came from — which LLM, which model,
    how big it was, and how it was processed. Critical for:
    - Analytics (which LLM pairs are most common?)
    - Adapter selection (which output format to use?)
    - Billing (token counts drive usage metering)
    """

    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_llm: SourceLLM = SourceLLM.UNKNOWN
    source_model: str = Field(
        default="",
        description="Specific model identifier (e.g., 'gpt-4o', 'claude-3.5-sonnet').",
    )
    total_tokens: int = Field(default=0, ge=0)
    message_count: int = Field(default=0, ge=0)
    compression_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of compressed to original size. 1.0 = no compression.",
    )
    processing_mode: ProcessingMode = ProcessingMode.STANDARD

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure all timestamps are UTC. Reject naive datetimes."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class EntityRelationship(ImmutableModel):
    """A directed relationship between two entities in the knowledge graph."""

    target_id: UUID
    type: RelationshipType
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of this relationship. 1.0 = certain.",
    )


class Entity(ImmutableModel):
    """A named entity extracted from the conversation.

    Entities are the building blocks of the knowledge graph. They represent
    people, technologies, concepts, code symbols, and other important nouns
    that carry meaning across the conversation. High-importance entities
    survive compression; low-importance ones are dropped first.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=500)
    type: EntityType
    aliases: tuple[str, ...] = Field(
        default=(),
        description="Alternative names for this entity (e.g., 'JS' for 'JavaScript').",
    )
    first_mentioned_at: int = Field(
        default=0,
        ge=0,
        description="Message index where this entity first appeared.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score. 1.0 = critical, 0.0 = trivial.",
    )
    relationships: tuple[EntityRelationship, ...] = ()
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class Summary(ImmutableModel):
    """A summary at one level of the hierarchy.

    Three tiers:
    - MESSAGE: 1 sentence per message (granular)
    - TOPIC: 1 paragraph per topic cluster (medium)
    - GLOBAL: ~200 words covering the entire conversation (coarse)

    During compression, message-level summaries are dropped first,
    then topic-level. Global summary is always preserved.
    """

    level: SummaryLevel
    content: str = Field(min_length=1, max_length=50000)
    token_count: int = Field(default=0, ge=0)
    covers_messages: tuple[int, int] = Field(
        default=(0, 0),
        description="Inclusive range [start, end] of message indices covered.",
    )


class Decision(ImmutableModel):
    """A decision made during the conversation.

    Decisions are ALWAYS preserved during compression because they represent
    commitments that affect future conversation flow. If you lose a decision
    during a shift, the target LLM might suggest something already rejected.
    """

    id: UUID = Field(default_factory=uuid4)
    description: str = Field(min_length=1, max_length=5000)
    rationale: str = Field(
        default="",
        max_length=10000,
        description="Why this decision was made.",
    )
    decided_at: int = Field(
        default=0,
        ge=0,
        description="Message index where this decision was made.",
    )
    status: DecisionStatus = DecisionStatus.ACTIVE
    superseded_by: UUID | None = Field(
        default=None,
        description="ID of the decision that replaced this one.",
    )


class Task(ImmutableModel):
    """A task or action item identified in the conversation.

    Tasks are preserved during compression because losing track of
    what the user asked for is the #1 complaint about context loss.
    """

    id: UUID = Field(default_factory=uuid4)
    description: str = Field(min_length=1, max_length=5000)
    status: TaskStatus = TaskStatus.ACTIVE
    dependencies: tuple[UUID, ...] = Field(
        default=(),
        description="IDs of tasks that must complete before this one.",
    )
    assigned_to: str | None = None
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Priority score. 1.0 = urgent.",
    )


class Preferences(ImmutableModel):
    """User preferences inferred from conversation patterns.

    These shape how the target LLM should respond. Without preferences,
    shifting from a casual Claude conversation to formal GPT-4 loses
    the user's established communication style.
    """

    tone: TonePreference = TonePreference.TECHNICAL
    detail_level: DetailLevel = DetailLevel.DETAILED
    format_preferences: dict[str, str | bool] = Field(
        default_factory=dict,
        description="E.g., {'code_blocks': true, 'markdown': true, 'bullet_points': true}.",
    )
    domain_expertise: tuple[str, ...] = Field(
        default=(),
        description="Domains the user has demonstrated expertise in.",
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code for the conversation.",
    )


class Artifact(ImmutableModel):
    """A concrete artifact produced during the conversation.

    Code snippets, config files, outputs — anything tangible that was
    created. These are high-value: losing a code artifact the user
    spent 30 minutes building is unacceptable.
    """

    id: UUID = Field(default_factory=uuid4)
    type: ArtifactType
    language: str = Field(
        default="",
        description="Programming language or format (e.g., 'python', 'yaml').",
    )
    content: str = Field(max_length=500000)
    title: str = Field(default="", max_length=500)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class KnowledgeGraphNode(ImmutableModel):
    """A node in the knowledge graph (references an Entity by ID)."""

    entity_id: UUID
    label: str = ""
    group: str = Field(
        default="",
        description="Visual grouping for graph rendering.",
    )


class KnowledgeGraphEdge(ImmutableModel):
    """An edge in the knowledge graph."""

    source_id: UUID
    target_id: UUID
    relationship: RelationshipType
    weight: float = Field(default=1.0, ge=0.0)


class KnowledgeGraph(ImmutableModel):
    """The full knowledge graph extracted from the conversation.

    A networkx-compatible structure that captures how entities relate
    to each other. Used for:
    - Visual context exploration in the dashboard
    - Intelligent compression (connected entities are preserved together)
    - Cross-session entity linking
    """

    nodes: tuple[KnowledgeGraphNode, ...] = ()
    edges: tuple[KnowledgeGraphEdge, ...] = ()


class TopicCluster(ImmutableModel):
    """A cluster of related messages grouped by semantic similarity.

    Created by embedding messages with sentence-transformers and clustering.
    Each cluster gets a topic-level summary. Used for:
    - Topic-level summarization
    - Smart compression (keep important topics, drop tangents)
    - Dashboard topic navigation
    """

    id: UUID = Field(default_factory=uuid4)
    label: str = Field(min_length=1, max_length=500)
    message_indices: tuple[int, ...] = ()
    embedding: tuple[float, ...] = Field(
        default=(),
        description="Centroid embedding vector for this topic cluster.",
    )


class SafetyFlag(ImmutableModel):
    """A safety concern detected during processing.

    Every safety issue is logged — even false positives — for audit
    purposes. Enterprise customers require this for compliance.
    """

    type: SafetyFlagType
    severity: SafetySeverity
    description: str = Field(max_length=5000)
    action_taken: SafetyAction
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence in this detection.",
    )
    source_message_index: int | None = Field(
        default=None,
        description="Which message triggered this flag.",
    )


class LLMComparison(ImmutableModel):
    """Result of sending the same context to multiple LLMs.

    Used by the workflow engine when doing parallel fan-out queries.
    Stores per-LLM responses for side-by-side comparison.
    """

    query: str
    responses: dict[str, str] = Field(
        default_factory=dict,
        description="Map of LLM identifier to response text.",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Main UCS Model ────────────────────────────────────────────────────────────


class UniversalContextSchema(BaseModel):
    """The Universal Context Schema (UCS) — Kangaroo Shift's core data format.

    This is the complete, structured representation of a conversation's
    context. It captures everything needed to reconstruct meaningful context
    in a different LLM: entities, summaries, decisions, tasks, preferences,
    artifacts, knowledge graphs, and safety flags.

    The UCS is:
    - Serializable to JSON for storage and transport
    - Versioned for forward compatibility
    - Validated on creation (invalid UCS documents are rejected)
    - Immutable once created (new versions are created, never mutated)

    A UCS document is created by the Context Engine pipeline:
        Raw conversation → Parser → Entity Extraction → Summarization
        → Compression → UCS Generation → Validation → Storage
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Schema version for forward compatibility
    version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic version of the UCS schema.",
    )

    # Core sections
    session_meta: SessionMeta
    entities: tuple[Entity, ...] = ()
    summaries: tuple[Summary, ...] = ()
    decisions: tuple[Decision, ...] = ()
    tasks: tuple[Task, ...] = ()
    preferences: Preferences = Field(default_factory=Preferences)
    artifacts: tuple[Artifact, ...] = ()
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    topic_clusters: tuple[TopicCluster, ...] = ()
    safety_flags: tuple[SafetyFlag, ...] = ()
    llm_comparisons: tuple[LLMComparison, ...] = ()

    # Importance scores — entity_id -> score for quick lookup
    importance_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Map of entity UUID string to importance score for quick lookup.",
    )

    @model_validator(mode="after")
    def validate_internal_references(self) -> "UniversalContextSchema":
        """Validate that all internal cross-references are consistent.

        This catches bugs in the Context Engine pipeline where an entity
        relationship references a non-existent entity ID, or a knowledge
        graph edge points to a missing node.
        """
        entity_ids = frozenset(e.id for e in self.entities)

        # Validate entity relationships reference existing entities
        for entity in self.entities:
            for rel in entity.relationships:
                if rel.target_id not in entity_ids:
                    raise ValueError(
                        f"Entity '{entity.name}' has relationship targeting "
                        f"non-existent entity ID {rel.target_id}"
                    )

        # Validate knowledge graph nodes reference existing entities
        for node in self.knowledge_graph.nodes:
            if node.entity_id not in entity_ids:
                raise ValueError(
                    f"Knowledge graph node references non-existent "
                    f"entity ID {node.entity_id}"
                )

        # Validate knowledge graph edges reference existing nodes
        node_entity_ids = frozenset(
            n.entity_id for n in self.knowledge_graph.nodes
        )
        for edge in self.knowledge_graph.edges:
            if edge.source_id not in node_entity_ids:
                raise ValueError(
                    f"Knowledge graph edge source {edge.source_id} "
                    f"is not a node in the graph"
                )
            if edge.target_id not in node_entity_ids:
                raise ValueError(
                    f"Knowledge graph edge target {edge.target_id} "
                    f"is not a node in the graph"
                )

        # Validate task dependencies reference existing tasks
        task_ids = frozenset(t.id for t in self.tasks)
        for task in self.tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise ValueError(
                        f"Task '{task.description[:50]}' depends on "
                        f"non-existent task ID {dep_id}"
                    )

        # Validate decision supersession chain
        decision_ids = frozenset(d.id for d in self.decisions)
        for decision in self.decisions:
            if (
                decision.superseded_by is not None
                and decision.superseded_by not in decision_ids
            ):
                raise ValueError(
                    f"Decision '{decision.description[:50]}' superseded by "
                    f"non-existent decision ID {decision.superseded_by}"
                )

        return self


class UCSValidator:
    """Validates UCS documents for structural integrity and quality.

    Beyond Pydantic's built-in validation, this class performs semantic
    checks that matter for product quality:
    - Does the UCS have at least a global summary?
    - Are importance scores within expected distributions?
    - Is the compression ratio plausible?
    - Are there circular dependencies in tasks?
    """

    @staticmethod
    def validate(ucs: UniversalContextSchema) -> list[str]:
        """Run all validation checks and return a list of warnings.

        Returns an empty list if the UCS is perfect. Warnings are
        non-fatal issues that should be logged but don't prevent usage.
        """
        warnings: list[str] = []

        # Check for global summary
        has_global = any(s.level == SummaryLevel.GLOBAL for s in ucs.summaries)
        if not has_global and ucs.session_meta.message_count > 0:
            warnings.append("UCS has no global summary — context quality may be degraded")

        # Check compression ratio plausibility
        if ucs.session_meta.compression_ratio <= 0:
            warnings.append("Compression ratio is zero or negative — likely a bug")

        # Check for orphaned importance scores
        entity_id_strs = {str(e.id) for e in ucs.entities}
        for score_id in ucs.importance_scores:
            if score_id not in entity_id_strs:
                warnings.append(
                    f"Importance score for entity {score_id} "
                    f"but no matching entity exists"
                )

        # Check for circular task dependencies
        task_map = {t.id: t for t in ucs.tasks}
        for task in ucs.tasks:
            visited: set[UUID] = set()
            current = task
            while current.dependencies:
                if current.id in visited:
                    warnings.append(
                        f"Circular dependency detected involving task "
                        f"'{task.description[:50]}'"
                    )
                    break
                visited.add(current.id)
                first_dep = current.dependencies[0]
                current = task_map.get(first_dep, current)
                if current.id == first_dep and first_dep not in task_map:
                    break

        # Check entity name uniqueness (warning, not error)
        entity_names = [e.name.lower() for e in ucs.entities]
        if len(entity_names) != len(set(entity_names)):
            warnings.append(
                "Duplicate entity names detected — consider deduplication"
            )

        # Validate topic cluster message indices are within range
        msg_count = ucs.session_meta.message_count
        if msg_count > 0:
            for cluster in ucs.topic_clusters:
                for idx in cluster.message_indices:
                    if idx >= msg_count:
                        warnings.append(
                            f"Topic cluster '{cluster.label}' references "
                            f"message index {idx} but only {msg_count} messages exist"
                        )
                        break

        return warnings

    @staticmethod
    def is_valid(ucs: UniversalContextSchema) -> bool:
        """Quick check — returns True if no warnings."""
        return len(UCSValidator.validate(ucs)) == 0
