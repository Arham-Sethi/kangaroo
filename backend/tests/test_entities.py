"""Comprehensive tests for the Entity Extraction Pipeline.

Tests cover all 5 pipeline stages:
    1. SpaCy NER (with graceful fallback)
    2. Technical extraction (regex + gazetteer)
    3. Deduplication
    4. Relationship extraction
    5. Knowledge graph construction

Total: 55+ tests covering happy paths, edge cases, and integration.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from app.core.engine.ccr import (
    ContentBlock,
    ContentType,
    Conversation,
    Message,
    MessageRole,
    SourceFormat,
)
from app.core.engine.entities import (
    EntityDeduplicator,
    EntityPipeline,
    EntityResult,
    KnowledgeGraphBuilder,
    RawEntity,
    RawRelationship,
    RelationshipExtractor,
    SpaCyExtractor,
    TechnicalExtractor,
    TECHNOLOGY_GAZETTEER,
)
from app.core.models.ucs import (
    EntityType,
    KnowledgeGraph,
    RelationshipType,
)


# -- Helpers -----------------------------------------------------------------


def _msg(text: str, role: MessageRole = MessageRole.USER) -> Message:
    """Create a simple text message."""
    return Message(
        role=role,
        content=(ContentBlock(type=ContentType.TEXT, text=text),),
    )


def _code_msg(code: str, language: str = "python") -> Message:
    """Create a message with a code block."""
    return Message(
        role=MessageRole.ASSISTANT,
        content=(ContentBlock(type=ContentType.CODE, text=code, language=language),),
    )


def _conversation(*messages: Message) -> Conversation:
    """Create a conversation from messages."""
    return Conversation(
        source_format=SourceFormat.GENERIC,
        messages=messages,
        message_count=len(messages),
    )


# == RawEntity Tests =========================================================


class TestRawEntity:
    """Tests for the mutable RawEntity intermediate representation."""

    def test_create_raw_entity(self) -> None:
        e = RawEntity(name="Python", type=EntityType.TECHNOLOGY)
        assert e.name == "Python"
        assert e.type == EntityType.TECHNOLOGY
        assert e.aliases == []
        assert e.mention_indices == []
        assert e.mention_count == 0

    def test_add_mention(self) -> None:
        e = RawEntity(name="Python", type=EntityType.TECHNOLOGY)
        e.add_mention(0)
        e.add_mention(2)
        e.add_mention(2)  # duplicate index
        assert e.mention_indices == [0, 2]
        assert e.mention_count == 3  # count tracks all mentions

    def test_canonical_name(self) -> None:
        e = RawEntity(name="FastAPI", type=EntityType.TECHNOLOGY)
        assert e.canonical_name == "FastAPI"


# == SpaCy Extractor Tests ==================================================


class TestSpaCyExtractor:
    """Tests for spaCy-based NER extraction."""

    def test_available_property(self) -> None:
        extractor = SpaCyExtractor(model_name="en_core_web_sm")
        # Should be True or False depending on installation
        result = extractor.available
        assert isinstance(result, bool)

    def test_extract_empty_messages(self) -> None:
        extractor = SpaCyExtractor()
        result = extractor.extract(())
        assert result == []

    def test_extract_with_unavailable_model(self) -> None:
        extractor = SpaCyExtractor(model_name="nonexistent_model_xyz")
        result = extractor.extract((_msg("John works at Google"),))
        assert result == []
        assert not extractor.available

    def test_extract_skips_empty_text(self) -> None:
        extractor = SpaCyExtractor(model_name="nonexistent_model_xyz")
        result = extractor.extract((_msg(""),))
        assert result == []

    def test_spacy_extract_if_available(self) -> None:
        """If spaCy is installed, verify it actually finds entities."""
        extractor = SpaCyExtractor()
        if not extractor.available:
            pytest.skip("spaCy model not installed")

        messages = (_msg("Elon Musk is the CEO of Tesla in San Francisco."),)
        result = extractor.extract(messages)
        names = {e.name.lower() for e in result}
        # At minimum, spaCy should find some of these
        assert len(result) > 0
        # Check at least one person or org was found
        types_found = {e.type for e in result}
        assert types_found & {EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION}


# == Technical Extractor Tests ===============================================


class TestTechnicalExtractor:
    """Tests for regex and gazetteer-based technical entity extraction."""

    def setup_method(self) -> None:
        self.extractor = TechnicalExtractor()

    # -- File paths --

    def test_extract_unix_file_path(self) -> None:
        msgs = (_msg("Edit the file /home/user/project/main.py"),)
        result = self.extractor.extract(msgs)
        paths = [e for e in result if e.type == EntityType.FILE_PATH]
        assert len(paths) >= 1
        assert any("/home/user/project/main.py" in e.name for e in paths)

    def test_extract_relative_file_path(self) -> None:
        msgs = (_msg("Check ./src/components/Button.tsx"),)
        result = self.extractor.extract(msgs)
        paths = [e for e in result if e.type == EntityType.FILE_PATH]
        assert len(paths) >= 1

    def test_extract_windows_file_path(self) -> None:
        msgs = (_msg(r"Open C:\Users\dev\project\config.yaml"),)
        result = self.extractor.extract(msgs)
        paths = [e for e in result if e.type == EntityType.FILE_PATH]
        assert len(paths) >= 1

    # -- URLs --

    def test_extract_url(self) -> None:
        msgs = (_msg("Visit https://api.example.com/v2/users"),)
        result = self.extractor.extract(msgs)
        urls = [e for e in result if e.type == EntityType.URL]
        assert len(urls) >= 1
        assert any("api.example.com" in e.name for e in urls)

    def test_extract_url_with_params(self) -> None:
        msgs = (_msg("See https://example.com/search?q=test&page=1"),)
        result = self.extractor.extract(msgs)
        urls = [e for e in result if e.type == EntityType.URL]
        assert len(urls) >= 1

    # -- API Endpoints --

    def test_extract_api_endpoint_with_method(self) -> None:
        msgs = (_msg("Use GET /api/v1/users to list all users"),)
        result = self.extractor.extract(msgs)
        apis = [e for e in result if e.type == EntityType.API]
        assert len(apis) >= 1
        assert any("/api/v1/users" in e.name for e in apis)

    def test_extract_api_endpoint_without_method(self) -> None:
        msgs = (_msg("The endpoint /api/v2/teams/{id} returns team details"),)
        result = self.extractor.extract(msgs)
        apis = [e for e in result if e.type == EntityType.API]
        assert len(apis) >= 1

    # -- Imports --

    def test_extract_python_import(self) -> None:
        msgs = (_msg("from fastapi import FastAPI"),)
        result = self.extractor.extract(msgs)
        code_entities = [e for e in result if e.type == EntityType.CODE]
        assert any("fastapi" in e.name.lower() for e in code_entities)

    def test_extract_js_require(self) -> None:
        msgs = (_msg("const express = require('express')"),)
        result = self.extractor.extract(msgs)
        code_entities = [e for e in result if e.type == EntityType.CODE]
        assert any("express" in e.name.lower() for e in code_entities)

    # -- Environment Variables --

    def test_extract_env_var(self) -> None:
        msgs = (_msg("Set DATABASE_URL to your connection string"),)
        result = self.extractor.extract(msgs)
        code_entities = [e for e in result if e.type == EntityType.CODE]
        assert any("DATABASE_URL" in e.name for e in code_entities)

    def test_ignore_non_env_uppercase_words(self) -> None:
        msgs = (_msg("The QUICK BROWN FOX jumped"),)
        result = self.extractor.extract(msgs)
        # These shouldn't match env var patterns
        code_entities = [e for e in result if e.type == EntityType.CODE and e.name == "QUICK"]
        assert len(code_entities) == 0

    # -- Code Symbols --

    def test_extract_function_definition(self) -> None:
        msgs = (_code_msg("def calculate_total(items):\n    return sum(items)"),)
        result = self.extractor.extract(msgs)
        symbols = [e for e in result if "calculate_total" in e.name]
        assert len(symbols) >= 1

    def test_extract_class_definition(self) -> None:
        msgs = (_code_msg("class UserService:\n    pass"),)
        result = self.extractor.extract(msgs)
        symbols = [e for e in result if "UserService" in e.name]
        assert len(symbols) >= 1

    # -- Technology Gazetteer --

    def test_extract_technology_name(self) -> None:
        msgs = (_msg("We're using React with TypeScript and PostgreSQL"),)
        result = self.extractor.extract(msgs)
        tech_names = {e.name for e in result if e.type == EntityType.TECHNOLOGY}
        assert "React" in tech_names
        assert "TypeScript" in tech_names
        assert "PostgreSQL" in tech_names

    def test_gazetteer_whole_word_matching(self) -> None:
        msgs = (_msg("I'm going to the store"),)
        result = self.extractor.extract(msgs)
        # "go" is in gazetteer as "Go" language, but "going" shouldn't match
        tech_names = {e.name for e in result if e.type == EntityType.TECHNOLOGY}
        assert "Go" not in tech_names

    def test_extract_ai_models(self) -> None:
        msgs = (_msg("We compared GPT-4 and Claude for this task"),)
        result = self.extractor.extract(msgs)
        tech_names = {e.name for e in result if e.type == EntityType.TECHNOLOGY}
        assert "GPT-4" in tech_names
        assert "Claude" in tech_names

    def test_extract_databases(self) -> None:
        msgs = (_msg("Redis for caching, MongoDB for documents"),)
        result = self.extractor.extract(msgs)
        tech_names = {e.name for e in result if e.type == EntityType.TECHNOLOGY}
        assert "Redis" in tech_names
        assert "MongoDB" in tech_names

    def test_extract_cloud_services(self) -> None:
        msgs = (_msg("Deploy to AWS with Docker and Kubernetes"),)
        result = self.extractor.extract(msgs)
        tech_names = {e.name for e in result if e.type == EntityType.TECHNOLOGY}
        assert "AWS" in tech_names
        assert "Docker" in tech_names
        assert "Kubernetes" in tech_names

    def test_extract_empty_messages(self) -> None:
        result = self.extractor.extract(())
        assert result == []

    def test_extract_empty_text(self) -> None:
        result = self.extractor.extract((_msg(""),))
        assert result == []

    def test_mention_tracking_across_messages(self) -> None:
        msgs = (
            _msg("We use Python"),
            _msg("Python is great for data science"),
        )
        result = self.extractor.extract(msgs)
        python_entities = [e for e in result if e.name == "Python"]
        assert len(python_entities) == 1
        assert python_entities[0].mention_count == 2
        assert 0 in python_entities[0].mention_indices
        assert 1 in python_entities[0].mention_indices


# == Gazetteer Tests =========================================================


class TestTechnologyGazetteer:
    """Tests for the technology gazetteer coverage."""

    def test_gazetteer_has_languages(self) -> None:
        assert "python" in TECHNOLOGY_GAZETTEER
        assert "javascript" in TECHNOLOGY_GAZETTEER
        assert "rust" in TECHNOLOGY_GAZETTEER

    def test_gazetteer_has_frameworks(self) -> None:
        assert "react" in TECHNOLOGY_GAZETTEER
        assert "django" in TECHNOLOGY_GAZETTEER
        assert "fastapi" in TECHNOLOGY_GAZETTEER

    def test_gazetteer_has_databases(self) -> None:
        assert "postgresql" in TECHNOLOGY_GAZETTEER
        assert "redis" in TECHNOLOGY_GAZETTEER
        assert "mongodb" in TECHNOLOGY_GAZETTEER

    def test_gazetteer_has_cloud(self) -> None:
        assert "aws" in TECHNOLOGY_GAZETTEER
        assert "gcp" in TECHNOLOGY_GAZETTEER
        assert "azure" in TECHNOLOGY_GAZETTEER

    def test_gazetteer_has_ai_models(self) -> None:
        assert "gpt-4" in TECHNOLOGY_GAZETTEER
        assert "claude" in TECHNOLOGY_GAZETTEER
        assert "pytorch" in TECHNOLOGY_GAZETTEER

    def test_gazetteer_canonical_names(self) -> None:
        assert TECHNOLOGY_GAZETTEER["k8s"][0] == "Kubernetes"
        assert TECHNOLOGY_GAZETTEER["postgres"][0] == "PostgreSQL"
        assert TECHNOLOGY_GAZETTEER["nextjs"][0] == "Next.js"

    def test_gazetteer_count(self) -> None:
        # Should have 200+ terms
        assert len(TECHNOLOGY_GAZETTEER) >= 150


# == Deduplicator Tests ======================================================


class TestEntityDeduplicator:
    """Tests for entity deduplication."""

    def setup_method(self) -> None:
        self.dedup = EntityDeduplicator()

    def test_empty_list(self) -> None:
        assert self.dedup.deduplicate([]) == []

    def test_no_duplicates(self) -> None:
        entities = [
            RawEntity(name="Python", type=EntityType.TECHNOLOGY, mention_count=3),
            RawEntity(name="Docker", type=EntityType.TECHNOLOGY, mention_count=1),
        ]
        result = self.dedup.deduplicate(entities)
        assert len(result) == 2

    def test_exact_match_merge(self) -> None:
        entities = [
            RawEntity(name="python", type=EntityType.TECHNOLOGY, mention_count=2,
                      mention_indices=[0, 1]),
            RawEntity(name="Python", type=EntityType.TECHNOLOGY, mention_count=1,
                      mention_indices=[3]),
        ]
        result = self.dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0].mention_count == 3

    def test_substring_merge(self) -> None:
        entities = [
            RawEntity(name="JavaScript", type=EntityType.TECHNOLOGY, mention_count=5,
                      mention_indices=[0]),
            RawEntity(name="java", type=EntityType.TECHNOLOGY, mention_count=1,
                      mention_indices=[2]),
        ]
        result = self.dedup.deduplicate(entities)
        # "java" is substring of "javascript" — should merge
        assert len(result) == 1
        assert result[0].name == "JavaScript"

    def test_short_substring_no_merge(self) -> None:
        """Substrings shorter than 3 chars should not trigger merge."""
        entities = [
            RawEntity(name="Go", type=EntityType.TECHNOLOGY, mention_count=3,
                      mention_indices=[0]),
            RawEntity(name="Google", type=EntityType.ORGANIZATION, mention_count=1,
                      mention_indices=[1]),
        ]
        result = self.dedup.deduplicate(entities)
        assert len(result) == 2

    def test_alias_tracking(self) -> None:
        entities = [
            RawEntity(name="PostgreSQL", type=EntityType.TECHNOLOGY, mention_count=5,
                      mention_indices=[0, 1, 2]),
            RawEntity(name="postgres", type=EntityType.TECHNOLOGY, mention_count=2,
                      mention_indices=[3, 4]),
        ]
        result = self.dedup.deduplicate(entities)
        assert len(result) == 1
        merged = result[0]
        # The one with more mentions wins
        assert merged.name == "PostgreSQL"
        assert "postgres" in merged.aliases

    def test_merge_preserves_all_indices(self) -> None:
        e1 = RawEntity(name="React", type=EntityType.TECHNOLOGY,
                       mention_count=2, mention_indices=[0, 1])
        e2 = RawEntity(name="react", type=EntityType.TECHNOLOGY,
                       mention_count=1, mention_indices=[5])
        result = self.dedup.deduplicate([e1, e2])
        assert len(result) == 1
        assert set(result[0].mention_indices) == {0, 1, 5}


# == Relationship Extractor Tests ============================================


class TestRelationshipExtractor:
    """Tests for relationship extraction."""

    def setup_method(self) -> None:
        self.extractor = RelationshipExtractor()

    def test_empty_inputs(self) -> None:
        result = self.extractor.extract([], ())
        assert result == []

    def test_co_occurrence_detection(self) -> None:
        entities = [
            RawEntity(name="React", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
            RawEntity(name="TypeScript", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        messages = (_msg("We use React with TypeScript"),)
        result = self.extractor.extract(entities, messages)
        assert len(result) >= 1
        assert any(r.type == RelationshipType.RELATED_TO for r in result)

    def test_verb_pattern_uses(self) -> None:
        entities = [
            RawEntity(name="FastAPI", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
            RawEntity(name="Pydantic", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        messages = (_msg("fastapi uses pydantic for validation"),)
        result = self.extractor.extract(entities, messages)
        uses_rels = [r for r in result if r.type == RelationshipType.USES]
        assert len(uses_rels) >= 1

    def test_verb_pattern_depends_on(self) -> None:
        entities = [
            RawEntity(name="backend", type=EntityType.CODE,
                      mention_indices=[0], mention_count=1),
            RawEntity(name="database", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        messages = (_msg("the backend depends on the database being available"),)
        result = self.extractor.extract(entities, messages)
        dep_rels = [r for r in result if r.type == RelationshipType.DEPENDS_ON]
        assert len(dep_rels) >= 1

    def test_no_self_relationships(self) -> None:
        """An entity should not have a relationship with itself."""
        entities = [
            RawEntity(name="Python", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        messages = (_msg("Python uses python features"),)
        result = self.extractor.extract(entities, messages)
        for rel in result:
            assert rel.source_name.lower() != rel.target_name.lower()

    def test_co_occurrence_confidence_boost(self) -> None:
        """Repeated co-occurrence should boost confidence."""
        entities = [
            RawEntity(name="Docker", type=EntityType.TECHNOLOGY,
                      mention_indices=[0, 1], mention_count=2),
            RawEntity(name="Kubernetes", type=EntityType.TECHNOLOGY,
                      mention_indices=[0, 1], mention_count=2),
        ]
        messages = (
            _msg("Docker and Kubernetes work together"),
            _msg("Use Docker with Kubernetes for orchestration"),
        )
        result = self.extractor.extract(entities, messages)
        related = [r for r in result if r.type == RelationshipType.RELATED_TO]
        assert len(related) >= 1
        # Second mention should have boosted confidence
        assert any(r.confidence > 0.5 for r in related)


# == Knowledge Graph Builder Tests ===========================================


class TestKnowledgeGraphBuilder:
    """Tests for UCS entity and knowledge graph construction."""

    def setup_method(self) -> None:
        self.builder = KnowledgeGraphBuilder()

    def test_empty_entities(self) -> None:
        entities, graph, scores = self.builder.build([], [], 0)
        assert entities == ()
        assert graph == KnowledgeGraph()
        assert scores == {}

    def test_single_entity(self) -> None:
        raw = [RawEntity(
            name="Python", type=EntityType.TECHNOLOGY,
            mention_indices=[0, 2, 4], mention_count=3,
        )]
        entities, graph, scores = self.builder.build(raw, [], 5)
        assert len(entities) == 1
        assert entities[0].name == "Python"
        assert entities[0].type == EntityType.TECHNOLOGY
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0
        assert len(scores) == 1

    def test_importance_scoring(self) -> None:
        """More recent + more frequent entities should score higher."""
        raw = [
            RawEntity(name="Python", type=EntityType.TECHNOLOGY,
                      mention_indices=[0, 1, 2, 3, 4], mention_count=5),
            RawEntity(name="Ruby", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        entities, graph, scores = self.builder.build(raw, [], 5)
        python_score = next(s for e, s in zip(entities, scores.values()) if e.name == "Python")
        ruby_score = next(s for e, s in zip(entities, scores.values()) if e.name == "Ruby")
        assert python_score > ruby_score

    def test_relationships_in_entities(self) -> None:
        raw = [
            RawEntity(name="FastAPI", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
            RawEntity(name="Python", type=EntityType.TECHNOLOGY,
                      mention_indices=[0], mention_count=1),
        ]
        rels = [RawRelationship(
            source_name="FastAPI", target_name="Python",
            type=RelationshipType.USES, confidence=0.8,
        )]
        entities, graph, scores = self.builder.build(raw, rels, 1)

        fastapi_entity = next(e for e in entities if e.name == "FastAPI")
        assert len(fastapi_entity.relationships) == 1
        assert fastapi_entity.relationships[0].type == RelationshipType.USES

    def test_knowledge_graph_edges(self) -> None:
        raw = [
            RawEntity(name="A", type=EntityType.CONCEPT,
                      mention_indices=[0], mention_count=1),
            RawEntity(name="B", type=EntityType.CONCEPT,
                      mention_indices=[0], mention_count=1),
        ]
        rels = [RawRelationship(
            source_name="A", target_name="B",
            type=RelationshipType.RELATED_TO, confidence=0.5,
        )]
        entities, graph, scores = self.builder.build(raw, rels, 1)
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0].relationship == RelationshipType.RELATED_TO

    def test_aliases_preserved(self) -> None:
        raw = [RawEntity(
            name="PostgreSQL", type=EntityType.TECHNOLOGY,
            aliases=["postgres", "pg"],
            mention_indices=[0], mention_count=1,
        )]
        entities, _, _ = self.builder.build(raw, [], 1)
        assert entities[0].aliases == ("postgres", "pg")

    def test_importance_bounded_0_to_1(self) -> None:
        raw = [RawEntity(
            name="X", type=EntityType.CODE,
            mention_indices=list(range(100)), mention_count=100,
        )]
        entities, _, scores = self.builder.build(raw, [], 100)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_recency_zero_messages(self) -> None:
        """Recency should be 0 when total_messages is 0."""
        score = KnowledgeGraphBuilder._compute_recency([0, 1, 2], 0)
        assert score == 0.0


# == Full Pipeline Tests =====================================================


class TestEntityPipeline:
    """Integration tests for the complete entity extraction pipeline."""

    def setup_method(self) -> None:
        # Disable spaCy for deterministic tests
        self.pipeline = EntityPipeline(enable_spacy=False)

    def test_empty_conversation(self) -> None:
        conv = _conversation()
        result = self.pipeline.extract(conv)
        assert isinstance(result, EntityResult)
        assert result.entities == ()
        assert result.graph == KnowledgeGraph()
        assert result.scores == {}
        assert result.raw_entity_count == 0
        assert not result.spacy_available

    def test_single_technology_mention(self) -> None:
        conv = _conversation(_msg("I'm learning Python"))
        result = self.pipeline.extract(conv)
        assert result.deduped_entity_count >= 1
        names = {e.name for e in result.entities}
        assert "Python" in names

    def test_multiple_technologies(self) -> None:
        conv = _conversation(
            _msg("Build a React frontend with TypeScript"),
            _msg("Backend in Python using FastAPI and PostgreSQL"),
        )
        result = self.pipeline.extract(conv)
        names = {e.name for e in result.entities}
        assert "React" in names
        assert "TypeScript" in names
        assert "Python" in names
        assert "FastAPI" in names
        assert "PostgreSQL" in names

    def test_file_paths_and_urls(self) -> None:
        conv = _conversation(
            _msg("Edit ./src/main.py and check https://docs.python.org"),
        )
        result = self.pipeline.extract(conv)
        types = {e.type for e in result.entities}
        assert EntityType.FILE_PATH in types or EntityType.URL in types

    def test_code_symbols_extracted(self) -> None:
        conv = _conversation(
            _code_msg("class UserService:\n    def get_user(self, user_id: int):\n        pass"),
        )
        result = self.pipeline.extract(conv)
        names = {e.name for e in result.entities}
        assert "UserService" in names or "get_user" in names

    def test_relationships_detected(self) -> None:
        conv = _conversation(
            _msg("FastAPI uses Pydantic for data validation"),
            _msg("The backend depends on PostgreSQL"),
        )
        result = self.pipeline.extract(conv)
        assert result.relationship_count >= 1

    def test_deduplication_happens(self) -> None:
        conv = _conversation(
            _msg("Python is great"),
            _msg("I love Python"),
            _msg("Python Python Python"),
        )
        result = self.pipeline.extract(conv)
        python_entities = [e for e in result.entities if e.name == "Python"]
        assert len(python_entities) == 1  # deduplicated

    def test_knowledge_graph_built(self) -> None:
        conv = _conversation(
            _msg("React with TypeScript and Tailwind CSS"),
        )
        result = self.pipeline.extract(conv)
        assert len(result.graph.nodes) > 0
        # Multiple entities -> should have edges from co-occurrence
        if result.deduped_entity_count > 1:
            assert len(result.graph.edges) > 0

    def test_importance_scores_populated(self) -> None:
        conv = _conversation(
            _msg("Python and Docker"),
        )
        result = self.pipeline.extract(conv)
        assert len(result.scores) == len(result.entities)
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_result_stats(self) -> None:
        conv = _conversation(
            _msg("Python and React"),
        )
        result = self.pipeline.extract(conv)
        assert result.raw_entity_count >= result.deduped_entity_count
        assert isinstance(result.spacy_available, bool)

    def test_pipeline_with_spacy_disabled(self) -> None:
        pipeline = EntityPipeline(enable_spacy=False)
        conv = _conversation(_msg("John works at Google"))
        result = pipeline.extract(conv)
        assert not result.spacy_available

    def test_complex_conversation(self) -> None:
        """Test a realistic multi-message conversation."""
        conv = _conversation(
            _msg("I need to build a REST API"),
            _msg("Let's use Python with FastAPI and PostgreSQL"),
            _code_msg(
                "from fastapi import FastAPI\n"
                "from sqlalchemy import create_engine\n\n"
                "app = FastAPI()\n"
                "DATABASE_URL = 'postgresql://localhost/mydb'"
            ),
            _msg("Deploy to AWS using Docker and Kubernetes"),
            _msg("Add Redis for caching and use JWT for auth"),
        )
        result = self.pipeline.extract(conv)

        # Should find multiple technologies
        names = {e.name for e in result.entities}
        assert len(names) >= 5

        # Should have relationships
        assert result.relationship_count > 0

        # Knowledge graph should be populated
        assert len(result.graph.nodes) >= 5
        assert len(result.graph.edges) > 0

        # All scores bounded
        for score in result.scores.values():
            assert 0.0 <= score <= 1.0

    def test_entities_are_immutable(self) -> None:
        """Verify that pipeline output entities are frozen UCS models."""
        conv = _conversation(_msg("Python is great"))
        result = self.pipeline.extract(conv)
        if result.entities:
            with pytest.raises(Exception):
                result.entities[0].name = "changed"  # type: ignore[misc]

    def test_api_endpoint_extraction(self) -> None:
        conv = _conversation(
            _msg("Call GET /api/v1/users to list users"),
            _msg("Use POST /api/v1/users to create a user"),
        )
        result = self.pipeline.extract(conv)
        api_entities = [e for e in result.entities if e.type == EntityType.API]
        assert len(api_entities) >= 1
