"""Git-like version control for UCS contexts.

Provides a DAG-based version history with branching, diffing, and history
traversal. Each node stores a UCS snapshot. Branches track head/base pointers.

This is an in-memory graph. For persistence, serialize via
VersionGraph.to_dict() / from_dict() and store in the database.

Usage:
    from app.core.storage.versioning import VersionGraph, compute_diff

    graph = VersionGraph()
    root = graph.create_root(ucs_v1)
    v2 = graph.commit(ucs_v2, parent_id=root.id)
    diff = graph.diff(root.id, v2.id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from app.core.models.ucs import UniversalContextSchema


# -- Enums -------------------------------------------------------------------


class BranchStatus(Enum):
    """Lifecycle states for a branch."""

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


# -- Data Structures ---------------------------------------------------------


@dataclass(frozen=True)
class VersionNode:
    """A single version in the DAG.

    Fields:
        id: Unique version identifier.
        parent_id: Parent version (None for root).
        branch_name: Which branch this commit belongs to.
        version_number: Sequential version number on this branch.
        ucs: The UCS snapshot at this version.
        message: Human-readable commit message.
        created_at: When this version was created.
        metadata: Arbitrary metadata (source, trigger, etc.).
    """

    id: UUID
    parent_id: UUID | None
    branch_name: str
    version_number: int
    ucs: UniversalContextSchema
    message: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Branch:
    """A named branch with head and base pointers.

    Fields:
        name: Branch name (e.g., "main", "experiment").
        head_id: Current tip of the branch.
        base_id: Where this branch was created from.
        status: Active, merged, or archived.
        created_at: When the branch was created.
    """

    name: str
    head_id: UUID
    base_id: UUID
    status: BranchStatus = BranchStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class VersionDiff:
    """Diff between two UCS versions.

    Fields:
        old_version_id: The older version.
        new_version_id: The newer version.
        entities_added: Entity names present in new but not old.
        entities_removed: Entity names present in old but not new.
        entities_modified: Entity names with changed importance.
        decisions_added: Count of new decisions.
        decisions_removed: Count of removed decisions.
        tasks_added: Count of new tasks.
        tasks_removed: Count of removed tasks.
        message_count_delta: Difference in message count.
        summary_changed: Whether the global summary changed.
    """

    old_version_id: UUID
    new_version_id: UUID
    entities_added: tuple[str, ...]
    entities_removed: tuple[str, ...]
    entities_modified: tuple[str, ...] = ()
    decisions_added: int = 0
    decisions_removed: int = 0
    tasks_added: int = 0
    tasks_removed: int = 0
    message_count_delta: int = 0
    summary_changed: bool = False


# -- Diff Function -----------------------------------------------------------


def compute_diff(
    old: UniversalContextSchema,
    new: UniversalContextSchema,
    old_id: UUID,
    new_id: UUID,
) -> VersionDiff:
    """Compute the diff between two UCS snapshots.

    Args:
        old: The older UCS version.
        new: The newer UCS version.
        old_id: Version ID of the old snapshot.
        new_id: Version ID of the new snapshot.

    Returns:
        VersionDiff describing what changed.
    """
    old_entity_names = {e.name for e in old.entities}
    new_entity_names = {e.name for e in new.entities}

    added = tuple(sorted(new_entity_names - old_entity_names))
    removed = tuple(sorted(old_entity_names - new_entity_names))

    shared = old_entity_names & new_entity_names
    old_importance = {e.name: e.importance for e in old.entities}
    new_importance = {e.name: e.importance for e in new.entities}
    modified = tuple(sorted(
        name for name in shared
        if old_importance.get(name) != new_importance.get(name)
    ))

    old_decisions = {d.description for d in old.decisions}
    new_decisions = {d.description for d in new.decisions}
    decisions_added = len(new_decisions - old_decisions)
    decisions_removed = len(old_decisions - new_decisions)

    old_tasks = {t.description for t in old.tasks}
    new_tasks = {t.description for t in new.tasks}
    tasks_added = len(new_tasks - old_tasks)
    tasks_removed = len(old_tasks - new_tasks)

    message_delta = (
        new.session_meta.message_count - old.session_meta.message_count
    )

    old_globals = [s.content for s in old.summaries if s.level.value == "global"]
    new_globals = [s.content for s in new.summaries if s.level.value == "global"]
    summary_changed = old_globals != new_globals

    return VersionDiff(
        old_version_id=old_id,
        new_version_id=new_id,
        entities_added=added,
        entities_removed=removed,
        entities_modified=modified,
        decisions_added=decisions_added,
        decisions_removed=decisions_removed,
        tasks_added=tasks_added,
        tasks_removed=tasks_removed,
        message_count_delta=message_delta,
        summary_changed=summary_changed,
    )


# -- Version Graph -----------------------------------------------------------


class VersionGraph:
    """In-memory DAG for UCS version history.

    Supports branching, linear commits, history traversal, and diffing.
    """

    def __init__(self, session_id: UUID | None = None) -> None:
        """Initialize an empty version graph.

        Args:
            session_id: Session this graph belongs to (auto-generated if omitted).
        """
        self._session_id = session_id or uuid4()
        self._nodes: dict[UUID, VersionNode] = {}
        self._branches: dict[str, Branch] = {}
        self._insert_order: list[UUID] = []

    @property
    def session_id(self) -> UUID:
        """The session this graph belongs to."""
        return self._session_id

    @property
    def node_count(self) -> int:
        """Number of versions in the graph."""
        return len(self._nodes)

    @property
    def branch_names(self) -> tuple[str, ...]:
        """Names of all branches."""
        return tuple(self._branches.keys())

    def create_root(
        self,
        ucs: UniversalContextSchema,
        *,
        branch: str = "main",
        message: str = "Initial version",
    ) -> VersionNode:
        """Create the root version (first commit).

        Args:
            ucs: The initial UCS snapshot.
            branch: Branch name for the root (default "main").
            message: Commit message.

        Returns:
            The root VersionNode.

        Raises:
            ValueError: If root already exists.
        """
        if self._nodes:
            raise ValueError("Root already exists. Use commit() for subsequent versions.")

        node_id = uuid4()
        node = VersionNode(
            id=node_id,
            parent_id=None,
            branch_name=branch,
            version_number=1,
            ucs=ucs,
            message=message,
        )
        self._nodes[node_id] = node
        self._insert_order.append(node_id)

        self._branches[branch] = Branch(
            name=branch,
            head_id=node_id,
            base_id=node_id,
        )

        return node

    def commit(
        self,
        ucs: UniversalContextSchema,
        *,
        parent_id: UUID,
        branch: str | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> VersionNode:
        """Create a new version as a child of parent_id.

        Args:
            ucs: The UCS snapshot for this version.
            parent_id: The parent version to build on.
            branch: Branch name (inherits from parent if omitted).
            message: Commit message.
            metadata: Arbitrary metadata.

        Returns:
            The new VersionNode.

        Raises:
            ValueError: If parent doesn't exist or branch is invalid.
        """
        if parent_id not in self._nodes:
            raise ValueError(f"Parent version not found: {parent_id}")

        parent = self._nodes[parent_id]
        target_branch = branch or parent.branch_name

        if target_branch not in self._branches:
            raise ValueError(
                f"Branch '{target_branch}' doesn't exist. "
                f"Create it first with create_branch()."
            )

        branch_obj = self._branches[target_branch]
        branch_versions = [
            n for n in self._nodes.values()
            if n.branch_name == target_branch
        ]
        next_version = max((n.version_number for n in branch_versions), default=0) + 1

        node_id = uuid4()
        node = VersionNode(
            id=node_id,
            parent_id=parent_id,
            branch_name=target_branch,
            version_number=next_version,
            ucs=ucs,
            message=message,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        self._insert_order.append(node_id)

        self._branches[target_branch] = Branch(
            name=target_branch,
            head_id=node_id,
            base_id=branch_obj.base_id,
            status=branch_obj.status,
            created_at=branch_obj.created_at,
        )

        return node

    def create_branch(
        self,
        name: str,
        *,
        from_id: UUID,
    ) -> Branch:
        """Create a new branch starting at a given version.

        Args:
            name: Branch name.
            from_id: Version to branch from.

        Returns:
            The new Branch.

        Raises:
            ValueError: If branch name exists or version not found.
        """
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists.")

        if from_id not in self._nodes:
            raise ValueError(f"Version not found: {from_id}")

        branch = Branch(
            name=name,
            head_id=from_id,
            base_id=from_id,
        )
        self._branches[name] = branch
        return branch

    def get_node(self, node_id: UUID) -> VersionNode:
        """Get a version node by ID.

        Raises:
            KeyError: If node not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Version not found: {node_id}")
        return self._nodes[node_id]

    def get_branch(self, name: str) -> Branch:
        """Get a branch by name.

        Raises:
            KeyError: If branch not found.
        """
        if name not in self._branches:
            raise KeyError(f"Branch not found: {name}")
        return self._branches[name]

    def get_branch_head(self, name: str) -> VersionNode:
        """Get the head node of a branch.

        Raises:
            KeyError: If branch not found.
        """
        branch = self.get_branch(name)
        return self._nodes[branch.head_id]

    def get_all_nodes(self) -> tuple[VersionNode, ...]:
        """Get all version nodes in insertion order."""
        return tuple(self._nodes[nid] for nid in self._insert_order)

    def get_history(
        self,
        node_id: UUID,
        max_depth: int | None = None,
    ) -> tuple[VersionNode, ...]:
        """Walk the parent chain from a node back to root.

        Args:
            node_id: Starting node.
            max_depth: Maximum number of nodes to return.

        Returns:
            Tuple of nodes from newest to oldest.

        Raises:
            KeyError: If node not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Version not found: {node_id}")

        history: list[VersionNode] = []
        current_id: UUID | None = node_id

        while current_id is not None:
            if max_depth is not None and len(history) >= max_depth:
                break
            node = self._nodes[current_id]
            history.append(node)
            current_id = node.parent_id

        return tuple(history)

    def diff(self, old_id: UUID, new_id: UUID) -> VersionDiff:
        """Compute diff between two versions in this graph.

        Args:
            old_id: The older version ID.
            new_id: The newer version ID.

        Returns:
            VersionDiff describing changes.

        Raises:
            KeyError: If either version not found.
        """
        old_node = self.get_node(old_id)
        new_node = self.get_node(new_id)
        return compute_diff(old_node.ucs, new_node.ucs, old_id, new_id)

    def archive_branch(self, name: str) -> Branch:
        """Archive a branch (cannot be main).

        Args:
            name: Branch to archive.

        Returns:
            The archived Branch.

        Raises:
            ValueError: If trying to archive main.
            KeyError: If branch not found.
        """
        if name == "main":
            raise ValueError("Cannot archive the main branch.")

        if name not in self._branches:
            raise KeyError(f"Branch not found: {name}")

        old = self._branches[name]
        archived = Branch(
            name=old.name,
            head_id=old.head_id,
            base_id=old.base_id,
            status=BranchStatus.ARCHIVED,
            created_at=old.created_at,
        )
        self._branches[name] = archived
        return archived
