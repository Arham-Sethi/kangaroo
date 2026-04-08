"""Memory consolidation — merge multiple UCS sessions into a unified brain.

The MemoryConsolidator takes N UCS documents from different sessions and
produces a single "brain" UCS that contains:
    - Deduplicated entities with merged importance scores
    - All unique decisions (with conflicts flagged)
    - Combined tasks (deduped by description similarity)
    - Merged preferences (most recent wins)
    - Unified knowledge graph

This is what makes Kangaroo a "Second Brain" — the ability to recall
across all conversations, not just the current one.

Usage:
    consolidator = MemoryConsolidator()
    brain_ucs = consolidator.consolidate(ucs_list)
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.core.brain.conflict import Conflict, ConflictDetector
from app.core.brain.temporal import DecayConfig, TemporalDecay
from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    Entity,
    EntityRelationship,
    EntityType,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    Preferences,
    SessionMeta,
    SourceLLM,
    Task,
    TaskStatus,
    UniversalContextSchema,
)


class MemoryConsolidator:
    """Merges multiple UCS sessions into a single unified brain UCS.

    Consolidation pipeline:
        1. Merge entities (deduplicate by name, combine importance)
        2. Merge decisions (keep all unique, flag conflicts)
        3. Merge tasks (deduplicate by description similarity)
        4. Merge preferences (most recent session wins)
        5. Build unified knowledge graph
    """

    def __init__(
        self,
        decay_config: DecayConfig | None = None,
        conflict_min_overlap: float = 0.1,
    ) -> None:
        self._decay = TemporalDecay(decay_config)
        self._conflict_detector = ConflictDetector(min_overlap=conflict_min_overlap)

    def consolidate(
        self,
        ucs_list: list[UniversalContextSchema],
        session_ages_days: list[float] | None = None,
    ) -> ConsolidationResult:
        """Consolidate multiple UCS documents into one brain.

        Args:
            ucs_list: UCS documents from different sessions.
            session_ages_days: Age in days for each session (for decay).
                              If None, all sessions are treated as current.

        Returns:
            ConsolidationResult with the unified UCS and metadata.
        """
        if not ucs_list:
            return ConsolidationResult(
                brain=_empty_brain(),
                session_count=0,
                entity_count=0,
                decision_count=0,
                task_count=0,
                conflicts=[],
                entities_merged=0,
            )

        ages = session_ages_days or [0.0] * len(ucs_list)

        # Step 1: Merge entities
        merged_entities, entities_merged = self._merge_entities(ucs_list, ages)

        # Step 2: Merge decisions
        merged_decisions = self._merge_decisions(ucs_list)

        # Step 3: Detect conflicts
        conflicts = self._conflict_detector.detect(ucs_list)

        # Step 4: Merge tasks
        merged_tasks = self._merge_tasks(ucs_list)

        # Step 5: Merge preferences (most recent session wins)
        merged_preferences = self._merge_preferences(ucs_list, ages)

        # Step 6: Build unified knowledge graph
        knowledge_graph = self._build_unified_graph(merged_entities)

        # Build the brain UCS
        total_messages = sum(u.session_meta.message_count for u in ucs_list)
        total_tokens = sum(u.session_meta.total_tokens for u in ucs_list)

        brain = UniversalContextSchema(
            session_meta=SessionMeta(
                source_llm=SourceLLM.UNKNOWN,
                source_model="kangaroo-brain",
                message_count=total_messages,
                total_tokens=total_tokens,
            ),
            entities=tuple(merged_entities),
            decisions=tuple(merged_decisions),
            tasks=tuple(merged_tasks),
            preferences=merged_preferences,
            knowledge_graph=knowledge_graph,
        )

        return ConsolidationResult(
            brain=brain,
            session_count=len(ucs_list),
            entity_count=len(merged_entities),
            decision_count=len(merged_decisions),
            task_count=len(merged_tasks),
            conflicts=conflicts,
            entities_merged=entities_merged,
        )

    # -- Entity merging ------------------------------------------------------

    def _merge_entities(
        self,
        ucs_list: list[UniversalContextSchema],
        ages: list[float],
    ) -> tuple[list[Entity], int]:
        """Deduplicate entities across sessions, apply temporal decay.

        Returns (merged_entities, count_of_merges).
        """
        # Index: canonical name -> (best entity, max importance, session indices)
        name_index: dict[str, _EntityAccum] = {}
        merge_count = 0

        for session_idx, ucs in enumerate(ucs_list):
            age = ages[session_idx]
            for entity in ucs.entities:
                # Apply temporal decay to importance
                decayed = self._decay.compute(
                    importance=entity.importance,
                    age_days=age,
                )

                key = entity.name.lower().strip()
                if key in name_index:
                    accum = name_index[key]
                    accum.merge(entity, decayed.decayed, session_idx)
                    merge_count += 1
                else:
                    name_index[key] = _EntityAccum(
                        entity=entity,
                        best_importance=decayed.decayed,
                        session_indices={session_idx},
                        aliases=set(entity.aliases),
                    )

                # Also check aliases for dedup
                for alias in entity.aliases:
                    alias_key = alias.lower().strip()
                    if alias_key != key and alias_key in name_index:
                        # This alias matches another entity — merge
                        accum = name_index[alias_key]
                        accum.add_alias(entity.name)
                        merge_count += 1

        # Build final entity list
        merged = []
        for accum in name_index.values():
            merged.append(accum.to_entity())

        # Sort by importance descending
        merged.sort(key=lambda e: e.importance, reverse=True)
        return merged, merge_count

    # -- Decision merging ----------------------------------------------------

    def _merge_decisions(
        self,
        ucs_list: list[UniversalContextSchema],
    ) -> list[Decision]:
        """Collect all unique decisions across sessions.

        Deduplication: decisions with identical descriptions (case-insensitive)
        are merged — the most recent status wins.
        """
        seen: dict[str, Decision] = {}

        for ucs in ucs_list:
            for decision in ucs.decisions:
                key = decision.description.lower().strip()
                if key not in seen:
                    seen[key] = decision
                else:
                    # If one is active and the other superseded, keep active
                    existing = seen[key]
                    if decision.status == DecisionStatus.ACTIVE:
                        seen[key] = decision
                    elif (
                        existing.status != DecisionStatus.ACTIVE
                        and decision.decided_at > existing.decided_at
                    ):
                        seen[key] = decision

        return list(seen.values())

    # -- Task merging --------------------------------------------------------

    def _merge_tasks(
        self,
        ucs_list: list[UniversalContextSchema],
    ) -> list[Task]:
        """Collect all unique tasks across sessions.

        Tasks with identical descriptions (case-insensitive) are deduped.
        Completed tasks from any session mark the merged task as completed.
        """
        seen: dict[str, Task] = {}

        for ucs in ucs_list:
            for task in ucs.tasks:
                key = task.description.lower().strip()
                if key not in seen:
                    seen[key] = task
                else:
                    existing = seen[key]
                    # If any session completed it, mark completed
                    if task.status == TaskStatus.COMPLETED:
                        seen[key] = task
                    elif (
                        existing.status != TaskStatus.COMPLETED
                        and task.priority > existing.priority
                    ):
                        seen[key] = task

        return list(seen.values())

    # -- Preference merging --------------------------------------------------

    def _merge_preferences(
        self,
        ucs_list: list[UniversalContextSchema],
        ages: list[float],
    ) -> Preferences:
        """Merge preferences — most recent session's preferences win.

        For format_preferences and domain_expertise, we union across sessions.
        """
        if not ucs_list:
            return Preferences()

        # Find the most recent session (smallest age)
        most_recent_idx = min(range(len(ages)), key=lambda i: ages[i])
        base_prefs = ucs_list[most_recent_idx].preferences

        # Union domain expertise from all sessions
        all_domains: set[str] = set()
        all_format_prefs: dict[str, str | bool] = {}

        for ucs in ucs_list:
            all_domains.update(ucs.preferences.domain_expertise)
            all_format_prefs.update(ucs.preferences.format_preferences)

        return Preferences(
            tone=base_prefs.tone,
            detail_level=base_prefs.detail_level,
            format_preferences=all_format_prefs,
            domain_expertise=tuple(sorted(all_domains)),
        )

    # -- Knowledge graph merging ---------------------------------------------

    def _build_unified_graph(
        self,
        entities: list[Entity],
    ) -> KnowledgeGraph:
        """Build a unified knowledge graph from merged entities."""
        nodes = []
        edges = []
        entity_ids = {e.id for e in entities}

        for entity in entities:
            nodes.append(
                KnowledgeGraphNode(
                    entity_id=entity.id,
                    label=entity.name,
                    group=entity.type.value,
                )
            )

            for rel in entity.relationships:
                if rel.target_id in entity_ids:
                    edges.append(
                        KnowledgeGraphEdge(
                            source_id=entity.id,
                            target_id=rel.target_id,
                            relationship=rel.type,
                            weight=entity.importance,
                        )
                    )

        return KnowledgeGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
        )


# -- Internal helpers --------------------------------------------------------


class _EntityAccum:
    """Accumulator for merging entities with the same name."""

    def __init__(
        self,
        entity: Entity,
        best_importance: float,
        session_indices: set[int],
        aliases: set[str],
    ) -> None:
        self.entity = entity
        self.best_importance = best_importance
        self.session_indices = session_indices
        self.aliases = aliases

    def merge(
        self,
        other: Entity,
        other_importance: float,
        session_idx: int,
    ) -> None:
        """Merge another entity into this accumulator."""
        self.session_indices.add(session_idx)
        self.aliases.update(other.aliases)
        self.aliases.add(other.name)

        if other_importance > self.best_importance:
            self.best_importance = other_importance
            self.entity = other

    def add_alias(self, name: str) -> None:
        self.aliases.add(name)

    def to_entity(self) -> Entity:
        """Build the final merged entity."""
        # Remove self-referencing alias
        final_aliases = self.aliases - {self.entity.name}

        return Entity(
            id=self.entity.id,
            name=self.entity.name,
            type=self.entity.type,
            aliases=tuple(sorted(final_aliases)),
            first_mentioned_at=self.entity.first_mentioned_at,
            importance=round(min(self.best_importance, 1.0), 4),
            relationships=self.entity.relationships,
            metadata={
                **self.entity.metadata,
                "session_count": len(self.session_indices),
            },
        )


class ConsolidationResult:
    """Result of memory consolidation.

    Attributes:
        brain: The unified UCS document.
        session_count: Number of sessions consolidated.
        entity_count: Entities in the brain.
        decision_count: Decisions in the brain.
        task_count: Tasks in the brain.
        conflicts: Detected decision conflicts.
        entities_merged: Number of entity merge operations performed.
    """

    __slots__ = (
        "brain",
        "session_count",
        "entity_count",
        "decision_count",
        "task_count",
        "conflicts",
        "entities_merged",
    )

    def __init__(
        self,
        brain: UniversalContextSchema,
        session_count: int,
        entity_count: int,
        decision_count: int,
        task_count: int,
        conflicts: list[Conflict],
        entities_merged: int,
    ) -> None:
        self.brain = brain
        self.session_count = session_count
        self.entity_count = entity_count
        self.decision_count = decision_count
        self.task_count = task_count
        self.conflicts = conflicts
        self.entities_merged = entities_merged


def _empty_brain() -> UniversalContextSchema:
    """Create an empty brain UCS."""
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.UNKNOWN,
            source_model="kangaroo-brain",
            message_count=0,
            total_tokens=0,
        ),
    )
