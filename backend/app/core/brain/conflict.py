"""Conflict detection — find contradictory decisions across sessions.

When a user makes decisions in separate sessions, they can contradict:
    - "Use REST" in session A vs "Use GraphQL" in session B
    - "Deploy on AWS" vs "Deploy on GCP"
    - "Use PostgreSQL" vs "Use MongoDB"

The ConflictDetector compares decisions across UCS documents using
keyword overlap and semantic similarity to surface contradictions.

Usage:
    detector = ConflictDetector()
    conflicts = detector.detect(ucs_list)
    for conflict in conflicts:
        print(f"{conflict.decision_a.description} vs {conflict.decision_b.description}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from app.core.models.ucs import Decision, DecisionStatus, UniversalContextSchema


@dataclass(frozen=True)
class Conflict:
    """A detected contradiction between two decisions.

    Attributes:
        decision_a: First decision.
        decision_b: Second decision (contradicts A).
        session_a_index: Index of the UCS containing decision A.
        session_b_index: Index of the UCS containing decision B.
        overlap_score: Keyword overlap score (0..1) — how related the topics are.
        conflict_type: Category of conflict (e.g., "opposing_choice", "scope_overlap").
        description: Human-readable explanation of the conflict.
    """

    decision_a: Decision
    decision_b: Decision
    session_a_index: int
    session_b_index: int
    overlap_score: float
    conflict_type: str
    description: str


# -- Known opposing technology/concept pairs ---------------------------------

_OPPOSING_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"rest", "restful"}, {"graphql"}),
    ({"sql", "postgresql", "postgres", "mysql", "sqlite"}, {"nosql", "mongodb", "dynamodb", "cassandra"}),
    ({"monolith", "monolithic"}, {"microservice", "microservices"}),
    ({"aws", "amazon"}, {"gcp", "google cloud"}),
    ({"aws", "amazon"}, {"azure", "microsoft cloud"}),
    ({"gcp", "google cloud"}, {"azure", "microsoft cloud"}),
    ({"react"}, {"vue", "vuejs"}),
    ({"react"}, {"angular"}),
    ({"vue", "vuejs"}, {"angular"}),
    ({"typescript"}, {"javascript"}),
    ({"python"}, {"go", "golang"}),
    ({"docker"}, {"podman"}),
    ({"kubernetes", "k8s"}, {"ecs", "fargate"}),
    ({"serverless", "lambda"}, {"containers", "kubernetes", "k8s"}),
    ({"jwt"}, {"session", "cookie"}),
    ({"ssr"}, {"spa", "csr"}),
]


def _tokenize(text: str) -> set[str]:
    """Extract lowercase tokens from text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard-like overlap between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _check_opposing_pairs(tokens_a: set[str], tokens_b: set[str]) -> bool:
    """Check if the two decisions reference known opposing technologies."""
    for side_x, side_y in _OPPOSING_PAIRS:
        a_has_x = bool(tokens_a & side_x)
        a_has_y = bool(tokens_a & side_y)
        b_has_x = bool(tokens_b & side_x)
        b_has_y = bool(tokens_b & side_y)
        # A chose X, B chose Y (or vice versa)
        if (a_has_x and b_has_y) or (a_has_y and b_has_x):
            return True
    return False


class ConflictDetector:
    """Detects contradictory decisions across multiple UCS sessions.

    Two decisions conflict when:
        1. They share significant keyword overlap (same topic), AND
        2. They reference known opposing choices, OR
        3. One supersedes the other explicitly.
    """

    def __init__(self, min_overlap: float = 0.1) -> None:
        """Initialize the detector.

        Args:
            min_overlap: Minimum keyword overlap to consider a potential conflict.
        """
        self._min_overlap = min_overlap

    def detect(
        self,
        ucs_list: list[UniversalContextSchema],
    ) -> list[Conflict]:
        """Detect conflicts across all pairs of UCS documents.

        Only compares active decisions (not superseded or reverted).

        Args:
            ucs_list: List of UCS documents from different sessions.

        Returns:
            List of detected conflicts, sorted by overlap score descending.
        """
        # Collect all active decisions with their session index
        indexed_decisions: list[tuple[int, Decision]] = []
        for idx, ucs in enumerate(ucs_list):
            for decision in ucs.decisions:
                if decision.status == DecisionStatus.ACTIVE:
                    indexed_decisions.append((idx, decision))

        conflicts: list[Conflict] = []
        seen_pairs: set[tuple[UUID, UUID]] = set()

        for i, (idx_a, dec_a) in enumerate(indexed_decisions):
            tokens_a = _tokenize(dec_a.description + " " + dec_a.rationale)

            for j in range(i + 1, len(indexed_decisions)):
                idx_b, dec_b = indexed_decisions[j]

                # Skip same-session comparisons
                if idx_a == idx_b:
                    continue

                # Skip already compared pairs
                pair_key = (
                    min(dec_a.id, dec_b.id),
                    max(dec_a.id, dec_b.id),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                tokens_b = _tokenize(dec_b.description + " " + dec_b.rationale)
                overlap = _keyword_overlap(tokens_a, tokens_b)

                # Check for explicit opposing technology pairs
                is_opposing = _check_opposing_pairs(tokens_a, tokens_b)

                if is_opposing or overlap >= self._min_overlap:
                    conflict_type = (
                        "opposing_choice" if is_opposing else "scope_overlap"
                    )
                    description = (
                        f"Session {idx_a} decided: '{dec_a.description}' "
                        f"vs Session {idx_b} decided: '{dec_b.description}'"
                    )

                    conflicts.append(
                        Conflict(
                            decision_a=dec_a,
                            decision_b=dec_b,
                            session_a_index=idx_a,
                            session_b_index=idx_b,
                            overlap_score=round(overlap, 4),
                            conflict_type=conflict_type,
                            description=description,
                        )
                    )

        # Sort by opposing_choice first, then by overlap score
        return sorted(
            conflicts,
            key=lambda c: (c.conflict_type != "opposing_choice", -c.overlap_score),
        )

    def detect_for_entity(
        self,
        ucs_list: list[UniversalContextSchema],
        entity_name: str,
    ) -> list[Conflict]:
        """Detect conflicts related to a specific entity.

        Filters to decisions that mention the entity name.
        """
        all_conflicts = self.detect(ucs_list)
        name_lower = entity_name.lower()
        return [
            c
            for c in all_conflicts
            if name_lower in c.decision_a.description.lower()
            or name_lower in c.decision_b.description.lower()
        ]
