"""Tests for ConflictDetector — find contradictory decisions across sessions.

Tests cover:
    - Opposing technology pair detection (REST vs GraphQL, etc.)
    - Keyword overlap scoring
    - Same-session decisions are NOT compared
    - Superseded decisions are excluded
    - No conflicts when topics differ
    - detect_for_entity filtering
    - Empty input handling
"""

import pytest

from app.core.brain.conflict import (
    Conflict,
    ConflictDetector,
    _check_opposing_pairs,
    _keyword_overlap,
    _tokenize,
)
from app.core.models.ucs import (
    Decision,
    DecisionStatus,
    Entity,
    EntityType,
    SessionMeta,
    SourceLLM,
    UniversalContextSchema,
)


# -- Helpers -----------------------------------------------------------------


def _make_ucs(
    decisions: list[Decision] | None = None,
) -> UniversalContextSchema:
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=5,
            total_tokens=500,
        ),
        decisions=tuple(decisions or []),
    )


def _decision(desc: str, rationale: str = "", status: DecisionStatus = DecisionStatus.ACTIVE) -> Decision:
    return Decision(description=desc, rationale=rationale, status=status)


# -- Unit tests for helpers --------------------------------------------------


class TestTokenize:
    def test_basic(self) -> None:
        tokens = _tokenize("Use REST API for the backend")
        assert "rest" in tokens
        assert "api" in tokens
        assert "backend" in tokens

    def test_lowercase(self) -> None:
        tokens = _tokenize("PostgreSQL")
        assert "postgresql" in tokens

    def test_empty(self) -> None:
        assert _tokenize("") == set()


class TestKeywordOverlap:
    def test_identical(self) -> None:
        tokens = {"rest", "api"}
        assert _keyword_overlap(tokens, tokens) == 1.0

    def test_no_overlap(self) -> None:
        assert _keyword_overlap({"rest"}, {"graphql"}) == 0.0

    def test_partial_overlap(self) -> None:
        a = {"use", "rest", "api"}
        b = {"use", "graphql", "api"}
        overlap = _keyword_overlap(a, b)
        assert 0 < overlap < 1.0

    def test_empty_sets(self) -> None:
        assert _keyword_overlap(set(), {"a"}) == 0.0
        assert _keyword_overlap({"a"}, set()) == 0.0


class TestOpposingPairs:
    def test_rest_vs_graphql(self) -> None:
        assert _check_opposing_pairs({"rest"}, {"graphql"}) is True

    def test_sql_vs_nosql(self) -> None:
        assert _check_opposing_pairs({"postgresql"}, {"mongodb"}) is True

    def test_monolith_vs_microservice(self) -> None:
        assert _check_opposing_pairs({"monolith"}, {"microservices"}) is True

    def test_same_side_not_opposing(self) -> None:
        assert _check_opposing_pairs({"rest"}, {"rest"}) is False

    def test_unrelated_not_opposing(self) -> None:
        assert _check_opposing_pairs({"python"}, {"docker"}) is False


# -- ConflictDetector tests --------------------------------------------------


class TestConflictDetector:
    def test_detect_opposing_decisions(self) -> None:
        ucs_a = _make_ucs([_decision("Use REST API for backend")])
        ucs_b = _make_ucs([_decision("Use GraphQL for backend")])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs_a, ucs_b])

        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == "opposing_choice"

    def test_no_conflict_same_decision(self) -> None:
        ucs_a = _make_ucs([_decision("Use Python for backend")])
        ucs_b = _make_ucs([_decision("Use Python for backend")])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs_a, ucs_b])

        # Same description — should still compare but might flag as scope_overlap
        # depending on keyword overlap. With identical text overlap is 1.0
        for c in conflicts:
            assert c.conflict_type != "opposing_choice"

    def test_no_conflict_different_topics(self) -> None:
        ucs_a = _make_ucs([_decision("Use blue color scheme for UI")])
        ucs_b = _make_ucs([_decision("Deploy on Kubernetes for scaling")])

        detector = ConflictDetector(min_overlap=0.3)
        conflicts = detector.detect([ucs_a, ucs_b])
        assert len(conflicts) == 0

    def test_same_session_not_compared(self) -> None:
        ucs = _make_ucs([
            _decision("Use REST API"),
            _decision("Use GraphQL API"),
        ])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs])
        # Both decisions are in the same session (index 0)
        assert len(conflicts) == 0

    def test_superseded_decisions_excluded(self) -> None:
        ucs_a = _make_ucs([
            _decision("Use REST", status=DecisionStatus.SUPERSEDED),
        ])
        ucs_b = _make_ucs([_decision("Use GraphQL")])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs_a, ucs_b])
        assert len(conflicts) == 0

    def test_multiple_conflicts(self) -> None:
        ucs_a = _make_ucs([
            _decision("Use REST for API"),
            _decision("Deploy on AWS with ECS"),
        ])
        ucs_b = _make_ucs([
            _decision("Use GraphQL for API"),
            _decision("Deploy on GCP with Kubernetes"),
        ])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs_a, ucs_b])
        assert len(conflicts) >= 2

    def test_empty_input(self) -> None:
        detector = ConflictDetector()
        assert detector.detect([]) == []

    def test_single_session_no_conflicts(self) -> None:
        ucs = _make_ucs([_decision("Use REST")])
        detector = ConflictDetector()
        assert detector.detect([ucs]) == []

    def test_conflict_has_description(self) -> None:
        ucs_a = _make_ucs([_decision("Use REST")])
        ucs_b = _make_ucs([_decision("Use GraphQL")])

        detector = ConflictDetector()
        conflicts = detector.detect([ucs_a, ucs_b])
        assert len(conflicts) >= 1
        assert "REST" in conflicts[0].description or "GraphQL" in conflicts[0].description


class TestConflictDetectorForEntity:
    def test_filter_by_entity_name(self) -> None:
        ucs_a = _make_ucs([_decision("Use REST API for user service")])
        ucs_b = _make_ucs([_decision("Use GraphQL API for user service")])

        detector = ConflictDetector()
        conflicts = detector.detect_for_entity([ucs_a, ucs_b], "API")
        assert len(conflicts) >= 1

    def test_filter_excludes_unrelated(self) -> None:
        ucs_a = _make_ucs([_decision("Use REST API")])
        ucs_b = _make_ucs([_decision("Use GraphQL API")])

        detector = ConflictDetector()
        conflicts = detector.detect_for_entity([ucs_a, ucs_b], "database")
        assert len(conflicts) == 0
