"""Adaptive context compression with priority queue and token budgets.

The Compression Engine takes processed context (entities, summaries,
decisions, tasks, artifacts) and fits it into a target token budget.
This is critical because every LLM has a context window limit — you
can't just dump 100K tokens of context into a 4K window.

Compression strategy (priority-based dropping):
    1. Message-level summaries are dropped FIRST (fine-grained, recoverable)
    2. Topic-level summaries are dropped NEXT (medium granularity)
    3. Low-importance entities are dropped
    4. Artifacts are truncated (keep first N lines of code)
    5. Decisions and tasks are NEVER dropped (critical for continuity)
    6. Global summary is NEVER dropped (last line of defense)

The compressor uses a priority queue where each item has a score:
    priority = base_weight * importance_multiplier

Items are dequeued lowest-priority-first for removal until the
token budget is met.

Usage:
    from app.core.engine.compressor import CompressionPipeline, CompressionResult

    pipeline = CompressionPipeline(target_tokens=4000)
    result = pipeline.compress(summaries, entities, decisions, tasks, artifacts)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.core.models.ucs import (
    Artifact,
    Decision,
    Entity,
    Summary,
    SummaryLevel,
    Task,
)


# -- Priority weights --------------------------------------------------------

class ContentCategory(str, Enum):
    """Categories of compressible content with base weights."""

    MESSAGE_SUMMARY = "message_summary"
    TOPIC_SUMMARY = "topic_summary"
    GLOBAL_SUMMARY = "global_summary"
    ENTITY = "entity"
    DECISION = "decision"
    TASK = "task"
    ARTIFACT = "artifact"


# Base weights: higher = harder to drop
_BASE_WEIGHTS: dict[ContentCategory, float] = {
    ContentCategory.MESSAGE_SUMMARY: 1.0,   # Dropped first
    ContentCategory.TOPIC_SUMMARY: 3.0,     # Dropped second
    ContentCategory.ENTITY: 2.0,            # Importance-weighted
    ContentCategory.ARTIFACT: 2.5,          # Truncated, not dropped
    ContentCategory.GLOBAL_SUMMARY: 10.0,   # Almost never dropped
    ContentCategory.TASK: 10.0,             # Never dropped
    ContentCategory.DECISION: 10.0,         # Never dropped
}


# -- Priority Queue Item -----------------------------------------------------


@dataclass
class PriorityItem:
    """An item in the compression priority queue.

    Lower priority = dropped first.
    """

    priority: float
    category: ContentCategory
    token_count: int
    index: int  # original position for stable ordering
    data: Any   # the actual UCS object

    def __lt__(self, other: "PriorityItem") -> bool:
        """Lower priority drops first. Break ties by index (FIFO)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.index > other.index  # newer items drop first


# -- Token estimation --------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses the ~4 chars per token heuristic, which is accurate enough
    for compression budgeting. Exact token counting would require a
    tokenizer (tiktoken/etc.) and add a dependency we don't need
    for approximate budgeting.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_entity_tokens(entity: Entity) -> int:
    """Estimate token cost of an entity in serialized form."""
    parts = [entity.name]
    if entity.aliases:
        parts.extend(entity.aliases)
    for rel in entity.relationships:
        parts.append(rel.type.value)
    return estimate_tokens(" ".join(parts))


def estimate_artifact_tokens(artifact: Artifact) -> int:
    """Estimate token cost of an artifact."""
    return estimate_tokens(artifact.content) + estimate_tokens(artifact.title)


# -- Compression Result ------------------------------------------------------


@dataclass(frozen=True)
class CompressionResult:
    """Immutable result from the compression pipeline.

    Attributes:
        summaries: Summaries that survived compression.
        entities: Entities that survived compression.
        decisions: All decisions (never dropped).
        tasks: All tasks (never dropped).
        artifacts: Artifacts (possibly truncated).
        total_tokens: Estimated total tokens after compression.
        original_tokens: Estimated tokens before compression.
        compression_ratio: Ratio of compressed to original (0-1).
        items_dropped: Count of items removed during compression.
    """

    summaries: tuple[Summary, ...]
    entities: tuple[Entity, ...]
    decisions: tuple[Decision, ...]
    tasks: tuple[Task, ...]
    artifacts: tuple[Artifact, ...]
    total_tokens: int
    original_tokens: int
    compression_ratio: float
    items_dropped: int


# -- Artifact Truncation -----------------------------------------------------


def truncate_artifact(artifact: Artifact, max_tokens: int) -> Artifact:
    """Truncate an artifact's content to fit within a token budget.

    Keeps the first N characters (preserving the most important part
    of code — usually the imports and class/function signatures).

    Args:
        artifact: Original artifact.
        max_tokens: Maximum allowed tokens.

    Returns:
        New Artifact with truncated content (immutable copy).
    """
    current_tokens = estimate_artifact_tokens(artifact)
    if current_tokens <= max_tokens:
        return artifact

    # Approximate character limit
    max_chars = max_tokens * 4
    truncated_content = artifact.content[:max_chars]

    # Try to cut at a line boundary
    last_newline = truncated_content.rfind("\n")
    if last_newline > max_chars // 2:
        truncated_content = truncated_content[:last_newline]

    truncated_content += "\n... [truncated]"

    return Artifact(
        id=artifact.id,
        type=artifact.type,
        language=artifact.language,
        content=truncated_content,
        title=artifact.title,
        metadata=artifact.metadata,
    )


# -- Main Pipeline -----------------------------------------------------------


class CompressionPipeline:
    """Adaptive context compression using priority-based dropping.

    Usage:
        pipeline = CompressionPipeline(target_tokens=4000)
        result = pipeline.compress(summaries, entities, decisions, tasks, artifacts)

    Strategy:
        1. Estimate total tokens for all content
        2. If under budget, return everything unchanged
        3. Build priority queue with all droppable items
        4. Drop lowest-priority items until within budget
        5. Truncate large artifacts
        6. Return surviving content with compression stats
    """

    def __init__(
        self,
        target_tokens: int = 4000,
        artifact_max_tokens: int = 500,
    ) -> None:
        """Initialize the compression pipeline.

        Args:
            target_tokens: Target token budget for compressed output.
            artifact_max_tokens: Maximum tokens per artifact after truncation.
        """
        self._target_tokens = target_tokens
        self._artifact_max_tokens = artifact_max_tokens

    def compress(
        self,
        summaries: tuple[Summary, ...] = (),
        entities: tuple[Entity, ...] = (),
        decisions: tuple[Decision, ...] = (),
        tasks: tuple[Task, ...] = (),
        artifacts: tuple[Artifact, ...] = (),
        importance_scores: dict[str, float] | None = None,
    ) -> CompressionResult:
        """Run the compression pipeline.

        Args:
            summaries: All summaries from the summarization pipeline.
            entities: All entities from the entity extraction pipeline.
            decisions: Extracted decisions.
            tasks: Extracted tasks.
            artifacts: Code/file artifacts.
            importance_scores: Entity ID -> importance score mapping.

        Returns:
            CompressionResult with surviving content and stats.
        """
        scores = importance_scores or {}

        # Step 1: Estimate original total tokens
        original_tokens = self._estimate_total(
            summaries, entities, decisions, tasks, artifacts,
        )

        # Step 2: If under budget, return everything
        if original_tokens <= self._target_tokens:
            return CompressionResult(
                summaries=summaries,
                entities=entities,
                decisions=decisions,
                tasks=tasks,
                artifacts=artifacts,
                total_tokens=original_tokens,
                original_tokens=original_tokens,
                compression_ratio=1.0,
                items_dropped=0,
            )

        # Step 3: Build priority queue for droppable items
        heap: list[PriorityItem] = []
        counter = 0

        # Add summaries (message and topic are droppable; global is not)
        for summary in summaries:
            if summary.level == SummaryLevel.MESSAGE:
                priority = _BASE_WEIGHTS[ContentCategory.MESSAGE_SUMMARY]
            elif summary.level == SummaryLevel.TOPIC:
                priority = _BASE_WEIGHTS[ContentCategory.TOPIC_SUMMARY]
            else:
                # Global summary — very high priority (don't drop)
                priority = _BASE_WEIGHTS[ContentCategory.GLOBAL_SUMMARY]

            heapq.heappush(heap, PriorityItem(
                priority=priority,
                category=(
                    ContentCategory.GLOBAL_SUMMARY if summary.level == SummaryLevel.GLOBAL
                    else ContentCategory.MESSAGE_SUMMARY if summary.level == SummaryLevel.MESSAGE
                    else ContentCategory.TOPIC_SUMMARY
                ),
                token_count=summary.token_count or estimate_tokens(summary.content),
                index=counter,
                data=summary,
            ))
            counter += 1

        # Add entities (weighted by importance)
        for entity in entities:
            importance = scores.get(str(entity.id), entity.importance)
            priority = _BASE_WEIGHTS[ContentCategory.ENTITY] * (0.5 + importance)
            heapq.heappush(heap, PriorityItem(
                priority=priority,
                category=ContentCategory.ENTITY,
                token_count=estimate_entity_tokens(entity),
                index=counter,
                data=entity,
            ))
            counter += 1

        # Add artifacts (truncatable)
        for artifact in artifacts:
            heapq.heappush(heap, PriorityItem(
                priority=_BASE_WEIGHTS[ContentCategory.ARTIFACT],
                category=ContentCategory.ARTIFACT,
                token_count=estimate_artifact_tokens(artifact),
                index=counter,
                data=artifact,
            ))
            counter += 1

        # Decisions and tasks are NOT added to the heap (never dropped)
        protected_tokens = sum(
            estimate_tokens(d.description + d.rationale) for d in decisions
        ) + sum(
            estimate_tokens(t.description) for t in tasks
        )

        # Step 4: Drop lowest-priority items until within budget
        current_tokens = original_tokens
        items_dropped = 0
        dropped_indices: set[int] = set()

        while current_tokens > self._target_tokens and heap:
            item = heapq.heappop(heap)

            # Don't drop protected categories
            if item.category in (
                ContentCategory.DECISION,
                ContentCategory.TASK,
                ContentCategory.GLOBAL_SUMMARY,
            ):
                # Put it back — we can't drop this
                heapq.heappush(heap, PriorityItem(
                    priority=item.priority + 100,  # bump so we don't loop
                    category=item.category,
                    token_count=item.token_count,
                    index=item.index,
                    data=item.data,
                ))
                # If only protected items remain, break
                if all(
                    i.category in (
                        ContentCategory.DECISION,
                        ContentCategory.TASK,
                        ContentCategory.GLOBAL_SUMMARY,
                    )
                    for i in heap
                ):
                    break
                continue

            # Truncate artifacts instead of dropping them entirely
            if item.category == ContentCategory.ARTIFACT:
                truncated = truncate_artifact(item.data, self._artifact_max_tokens)
                saved = item.token_count - estimate_artifact_tokens(truncated)
                if saved > 0:
                    current_tokens -= saved
                    # Replace with truncated version
                    heapq.heappush(heap, PriorityItem(
                        priority=item.priority + 50,  # don't re-truncate
                        category=item.category,
                        token_count=estimate_artifact_tokens(truncated),
                        index=item.index,
                        data=truncated,
                    ))
                    continue

            # Drop the item
            dropped_indices.add(item.index)
            current_tokens -= item.token_count
            items_dropped += 1

        # Step 5: Collect surviving items
        surviving_summaries: list[Summary] = []
        surviving_entities: list[Entity] = []
        surviving_artifacts: list[Artifact] = []

        # Collect from heap + check against dropped set
        remaining_items: list[PriorityItem] = list(heap)
        for item in remaining_items:
            if item.index in dropped_indices:
                continue
            if item.category in (
                ContentCategory.MESSAGE_SUMMARY,
                ContentCategory.TOPIC_SUMMARY,
                ContentCategory.GLOBAL_SUMMARY,
            ):
                surviving_summaries.append(item.data)
            elif item.category == ContentCategory.ENTITY:
                surviving_entities.append(item.data)
            elif item.category == ContentCategory.ARTIFACT:
                surviving_artifacts.append(item.data)

        # Recalculate actual total
        final_tokens = self._estimate_total(
            tuple(surviving_summaries),
            tuple(surviving_entities),
            decisions,
            tasks,
            tuple(surviving_artifacts),
        )

        ratio = final_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressionResult(
            summaries=tuple(surviving_summaries),
            entities=tuple(surviving_entities),
            decisions=decisions,
            tasks=tasks,
            artifacts=tuple(surviving_artifacts),
            total_tokens=final_tokens,
            original_tokens=original_tokens,
            compression_ratio=round(ratio, 4),
            items_dropped=items_dropped,
        )

    @staticmethod
    def _estimate_total(
        summaries: tuple[Summary, ...],
        entities: tuple[Entity, ...],
        decisions: tuple[Decision, ...],
        tasks: tuple[Task, ...],
        artifacts: tuple[Artifact, ...],
    ) -> int:
        """Estimate total token count for all content."""
        total = 0
        for s in summaries:
            total += s.token_count or estimate_tokens(s.content)
        for e in entities:
            total += estimate_entity_tokens(e)
        for d in decisions:
            total += estimate_tokens(d.description + d.rationale)
        for t in tasks:
            total += estimate_tokens(t.description)
        for a in artifacts:
            total += estimate_artifact_tokens(a)
        return total
