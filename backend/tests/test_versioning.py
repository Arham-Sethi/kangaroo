"""Tests for git-like context versioning and branching.

Tests cover:
    - Version graph creation (root, commits)
    - Branching and branch management
    - Version history traversal
    - Diffing between versions
    - Branch archival
    - Error handling
"""

from uuid import uuid4

import pytest

from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    Entity,
    EntityType,
    SessionMeta,
    SourceLLM,
    Task,
    TaskStatus,
    UniversalContextSchema,
)
from app.core.storage.versioning import (
    Branch,
    BranchStatus,
    VersionDiff,
    VersionGraph,
    VersionNode,
    compute_diff,
)


# -- Helpers -----------------------------------------------------------------


def _make_ucs(
    message_count: int = 3,
    entity_names: tuple[str, ...] = (),
    decision_descs: tuple[str, ...] = (),
    task_descs: tuple[str, ...] = (),
    entities: tuple[Entity, ...] | None = None,
    decisions: tuple[Decision, ...] | None = None,
    tasks: tuple[Task, ...] | None = None,
) -> UniversalContextSchema:
    """Create a UCS with configurable content for diff testing."""
    if entities is None:
        entities = tuple(
            Entity(name=name, type=EntityType.TECHNOLOGY, importance=0.8)
            for name in entity_names
        )
    if decisions is None:
        decisions = tuple(
            Decision(description=desc, rationale="test", status=DecisionStatus.ACTIVE)
            for desc in decision_descs
        )
    if tasks is None:
        tasks = tuple(
            Task(description=desc, status=TaskStatus.ACTIVE)
            for desc in task_descs
        )
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=message_count,
            total_tokens=message_count * 100,
        ),
        entities=entities,
        decisions=decisions,
        tasks=tasks,
    )


# == Version Graph Creation ==================================================


class TestVersionGraphCreation:
    """Tests for creating version graphs."""

    def test_create_empty_graph(self) -> None:
        graph = VersionGraph()
        assert graph.node_count == 0
        assert graph.branch_names == ()

    def test_create_graph_with_session_id(self) -> None:
        sid = uuid4()
        graph = VersionGraph(session_id=sid)
        assert graph.session_id == sid

    def test_create_graph_auto_session_id(self) -> None:
        graph = VersionGraph()
        assert graph.session_id is not None

    def test_create_root(self) -> None:
        graph = VersionGraph()
        ucs = _make_ucs()
        root = graph.create_root(ucs)
        assert isinstance(root, VersionNode)
        assert root.parent_id is None
        assert root.branch_name == "main"
        assert root.version_number == 1
        assert root.message == "Initial version"
        assert graph.node_count == 1

    def test_create_root_custom_branch(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs(), branch="develop")
        assert root.branch_name == "develop"
        assert "develop" in graph.branch_names

    def test_create_root_with_message(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs(), message="First context capture")
        assert root.message == "First context capture"

    def test_create_root_twice_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(ValueError, match="Root already exists"):
            graph.create_root(_make_ucs())

    def test_root_creates_main_branch(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        assert "main" in graph.branch_names
        branch = graph.get_branch("main")
        assert branch.status == BranchStatus.ACTIVE


# == Commits =================================================================


class TestCommits:
    """Tests for creating new versions."""

    @pytest.fixture
    def graph_with_root(self) -> tuple[VersionGraph, VersionNode]:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs(message_count=1))
        return graph, root

    def test_commit_creates_new_version(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(_make_ucs(message_count=5), parent_id=root.id)
        assert v2.parent_id == root.id
        assert v2.version_number == 2
        assert graph.node_count == 2

    def test_commit_increments_version_number(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(_make_ucs(), parent_id=root.id)
        v3 = graph.commit(_make_ucs(), parent_id=v2.id)
        assert v2.version_number == 2
        assert v3.version_number == 3

    def test_commit_inherits_branch(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(_make_ucs(), parent_id=root.id)
        assert v2.branch_name == "main"

    def test_commit_with_message(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(
            _make_ucs(),
            parent_id=root.id,
            message="Added more entities",
        )
        assert v2.message == "Added more entities"

    def test_commit_updates_branch_head(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(_make_ucs(), parent_id=root.id)
        head = graph.get_branch_head("main")
        assert head.id == v2.id

    def test_commit_with_invalid_parent_fails(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, _ = graph_with_root
        with pytest.raises(ValueError, match="Parent version not found"):
            graph.commit(_make_ucs(), parent_id=uuid4())

    def test_commit_with_nonexistent_branch_fails(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        with pytest.raises(ValueError, match="doesn't exist"):
            graph.commit(_make_ucs(), parent_id=root.id, branch="nonexistent")

    def test_commit_with_metadata(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        v2 = graph.commit(
            _make_ucs(),
            parent_id=root.id,
            metadata={"source": "api"},
        )
        assert v2.metadata["source"] == "api"


# == Branching ===============================================================


class TestBranching:
    """Tests for branch management."""

    @pytest.fixture
    def graph_with_root(self) -> tuple[VersionGraph, VersionNode]:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        return graph, root

    def test_create_branch(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        branch = graph.create_branch("experiment", from_id=root.id)
        assert isinstance(branch, Branch)
        assert branch.name == "experiment"
        assert branch.head_id == root.id
        assert branch.base_id == root.id
        assert branch.status == BranchStatus.ACTIVE

    def test_create_branch_appears_in_names(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        graph.create_branch("feature", from_id=root.id)
        assert "feature" in graph.branch_names
        assert "main" in graph.branch_names

    def test_commit_to_branch(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        graph.create_branch("experiment", from_id=root.id)
        v2 = graph.commit(
            _make_ucs(message_count=10),
            parent_id=root.id,
            branch="experiment",
        )
        assert v2.branch_name == "experiment"
        assert graph.get_branch_head("experiment").id == v2.id

    def test_parallel_branches(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        graph.create_branch("branch-a", from_id=root.id)
        graph.create_branch("branch-b", from_id=root.id)
        va = graph.commit(_make_ucs(message_count=5), parent_id=root.id, branch="branch-a")
        vb = graph.commit(_make_ucs(message_count=8), parent_id=root.id, branch="branch-b")
        assert graph.get_branch_head("branch-a").id == va.id
        assert graph.get_branch_head("branch-b").id == vb.id
        assert graph.get_branch_head("main").id == root.id

    def test_duplicate_branch_name_fails(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, root = graph_with_root
        graph.create_branch("feature", from_id=root.id)
        with pytest.raises(ValueError, match="already exists"):
            graph.create_branch("feature", from_id=root.id)

    def test_branch_from_nonexistent_version_fails(
        self, graph_with_root: tuple[VersionGraph, VersionNode],
    ) -> None:
        graph, _ = graph_with_root
        with pytest.raises(ValueError, match="Version not found"):
            graph.create_branch("bad", from_id=uuid4())


# == History =================================================================


class TestHistory:
    """Tests for version history traversal."""

    def test_root_history_is_single_node(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        history = graph.get_history(root.id)
        assert len(history) == 1
        assert history[0].id == root.id

    def test_linear_history(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        v2 = graph.commit(_make_ucs(), parent_id=root.id)
        v3 = graph.commit(_make_ucs(), parent_id=v2.id)
        history = graph.get_history(v3.id)
        assert len(history) == 3
        assert history[0].id == v3.id
        assert history[1].id == v2.id
        assert history[2].id == root.id

    def test_history_max_depth(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        current = root
        for _ in range(10):
            current = graph.commit(_make_ucs(), parent_id=current.id)
        history = graph.get_history(current.id, max_depth=5)
        assert len(history) == 5

    def test_history_nonexistent_node_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(KeyError, match="Version not found"):
            graph.get_history(uuid4())


# == Diffing =================================================================


class TestDiffing:
    """Tests for version diff computation."""

    def test_diff_no_changes(self) -> None:
        ucs = _make_ucs(entity_names=("Python",))
        diff = compute_diff(ucs, ucs, uuid4(), uuid4())
        assert diff.entities_added == ()
        assert diff.entities_removed == ()
        assert diff.entities_modified == ()
        assert diff.decisions_added == 0
        assert diff.tasks_added == 0
        assert diff.summary_changed is False
        assert diff.message_count_delta == 0

    def test_diff_entities_added(self) -> None:
        shared = Entity(name="Python", type=EntityType.TECHNOLOGY, importance=0.8)
        extra = Entity(name="FastAPI", type=EntityType.TECHNOLOGY, importance=0.8)
        old = _make_ucs(entities=(shared,))
        new = _make_ucs(entities=(shared, extra))
        diff = compute_diff(old, new, uuid4(), uuid4())
        assert len(diff.entities_added) == 1
        assert len(diff.entities_removed) == 0

    def test_diff_entities_removed(self) -> None:
        shared = Entity(name="Python", type=EntityType.TECHNOLOGY, importance=0.8)
        extra = Entity(name="FastAPI", type=EntityType.TECHNOLOGY, importance=0.8)
        old = _make_ucs(entities=(shared, extra))
        new = _make_ucs(entities=(shared,))
        diff = compute_diff(old, new, uuid4(), uuid4())
        assert len(diff.entities_removed) == 1
        assert len(diff.entities_added) == 0

    def test_diff_decisions_added(self) -> None:
        shared = Decision(description="Use PostgreSQL", rationale="test", status=DecisionStatus.ACTIVE)
        extra = Decision(description="Use Redis", rationale="test", status=DecisionStatus.ACTIVE)
        old = _make_ucs(decisions=(shared,))
        new = _make_ucs(decisions=(shared, extra))
        diff = compute_diff(old, new, uuid4(), uuid4())
        assert diff.decisions_added == 1

    def test_diff_tasks_added(self) -> None:
        shared = Task(description="Build API", status=TaskStatus.ACTIVE)
        extra = Task(description="Write tests", status=TaskStatus.ACTIVE)
        old = _make_ucs(tasks=(shared,))
        new = _make_ucs(tasks=(shared, extra))
        diff = compute_diff(old, new, uuid4(), uuid4())
        assert diff.tasks_added == 1

    def test_diff_message_count_delta(self) -> None:
        old = _make_ucs(message_count=5)
        new = _make_ucs(message_count=10)
        diff = compute_diff(old, new, uuid4(), uuid4())
        assert diff.message_count_delta == 5

    def test_diff_via_graph(self) -> None:
        graph = VersionGraph()
        shared = Entity(name="Python", type=EntityType.TECHNOLOGY, importance=0.8)
        extra = Entity(name="FastAPI", type=EntityType.TECHNOLOGY, importance=0.8)
        ucs1 = _make_ucs(entities=(shared,), message_count=3)
        ucs2 = _make_ucs(entities=(shared, extra), message_count=5)
        root = graph.create_root(ucs1)
        v2 = graph.commit(ucs2, parent_id=root.id)
        diff = graph.diff(root.id, v2.id)
        assert len(diff.entities_added) == 1
        assert diff.message_count_delta == 2

    def test_diff_nonexistent_version_fails(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        with pytest.raises(KeyError):
            graph.diff(root.id, uuid4())


# == Branch Archival =========================================================


class TestBranchArchival:
    """Tests for archiving branches."""

    def test_archive_branch(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        graph.create_branch("experiment", from_id=root.id)
        archived = graph.archive_branch("experiment")
        assert archived.status == BranchStatus.ARCHIVED

    def test_archive_main_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(ValueError, match="Cannot archive the main branch"):
            graph.archive_branch("main")

    def test_archive_nonexistent_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(KeyError, match="Branch not found"):
            graph.archive_branch("nonexistent")


# == Node Retrieval ==========================================================


class TestNodeRetrieval:
    """Tests for getting nodes and branches."""

    def test_get_node(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        node = graph.get_node(root.id)
        assert node.id == root.id

    def test_get_nonexistent_node_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(KeyError):
            graph.get_node(uuid4())

    def test_get_branch(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        branch = graph.get_branch("main")
        assert branch.name == "main"

    def test_get_nonexistent_branch_fails(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        with pytest.raises(KeyError):
            graph.get_branch("nonexistent")

    def test_get_branch_head(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        head = graph.get_branch_head("main")
        assert head.id == root.id

    def test_get_all_nodes(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        v2 = graph.commit(_make_ucs(), parent_id=root.id)
        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == 2
        assert all_nodes[0].id == root.id
        assert all_nodes[1].id == v2.id


# == Data Structures =========================================================


class TestDataStructures:
    """Tests for frozen dataclasses."""

    def test_version_node_is_frozen(self) -> None:
        graph = VersionGraph()
        root = graph.create_root(_make_ucs())
        with pytest.raises(AttributeError):
            root.message = "hacked"  # type: ignore[misc]

    def test_branch_is_frozen(self) -> None:
        graph = VersionGraph()
        graph.create_root(_make_ucs())
        branch = graph.get_branch("main")
        with pytest.raises(AttributeError):
            branch.name = "hacked"  # type: ignore[misc]

    def test_version_diff_is_frozen(self) -> None:
        diff = compute_diff(_make_ucs(), _make_ucs(), uuid4(), uuid4())
        with pytest.raises(AttributeError):
            diff.message_count_delta = 999  # type: ignore[misc]

    def test_branch_status_values(self) -> None:
        assert BranchStatus.ACTIVE.value == "active"
        assert BranchStatus.MERGED.value == "merged"
        assert BranchStatus.ARCHIVED.value == "archived"
