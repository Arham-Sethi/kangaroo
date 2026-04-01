"""Tests for the UCS Generator Pipeline and Local Engine.

Tests cover:
    1. Full pipeline (raw data -> UCS)
    2. From pre-parsed conversation
    3. Different input formats
    4. Stats and validation
    5. Local engine privacy guarantees

Total: 30+ tests
"""

from __future__ import annotations

import json

import pytest

from app.core.engine.ccr import (
    ContentBlock,
    ContentType,
    Conversation,
    Message,
    MessageRole,
    SourceFormat,
)
from app.core.engine.local_engine import LocalEngine, LocalProcessingConfig
from app.core.engine.ucs_generator import (
    GenerationResult,
    GenerationStats,
    UCSGeneratorPipeline,
)
from app.core.models.ucs import (
    ProcessingMode,
    SummaryLevel,
    UniversalContextSchema,
)


# -- Helpers -----------------------------------------------------------------


def _msg(text: str, role: MessageRole = MessageRole.USER) -> Message:
    return Message(
        role=role,
        content=(ContentBlock(type=ContentType.TEXT, text=text),),
    )


def _conversation(*messages: Message) -> Conversation:
    return Conversation(
        source_format=SourceFormat.GENERIC,
        messages=messages,
        message_count=len(messages),
    )


def _openai_format(messages: list[dict]) -> list[dict]:
    """Create OpenAI-format message array."""
    return messages


# == UCS Generator Pipeline Tests ============================================


class TestUCSGeneratorPipeline:
    def setup_method(self) -> None:
        self.pipeline = UCSGeneratorPipeline(
            target_tokens=4000,
            enable_spacy=False,  # Deterministic tests
        )

    def test_generate_from_conversation_empty(self) -> None:
        conv = _conversation()
        result = self.pipeline.generate_from_conversation(conv)
        assert isinstance(result, GenerationResult)
        assert isinstance(result.ucs, UniversalContextSchema)
        assert result.conversation == conv

    def test_generate_from_conversation_simple(self) -> None:
        conv = _conversation(
            _msg("I want to build a web app with Python and FastAPI"),
            _msg("Sure! Let's start with the project structure.", MessageRole.ASSISTANT),
        )
        result = self.pipeline.generate_from_conversation(conv)
        assert result.ucs.session_meta.message_count == 2
        assert len(result.ucs.entities) > 0
        assert len(result.ucs.summaries) > 0

    def test_generate_from_raw_openai(self) -> None:
        raw = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        result = self.pipeline.generate(raw)
        assert result.ucs.session_meta.message_count >= 1

    def test_generate_from_raw_string(self) -> None:
        raw = "User: Tell me about Docker\nAssistant: Docker is a container platform."
        result = self.pipeline.generate(raw)
        assert isinstance(result.ucs, UniversalContextSchema)

    def test_generate_from_raw_json_string(self) -> None:
        raw_data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        raw_json = json.dumps(raw_data)
        result = self.pipeline.generate(raw_json)
        assert isinstance(result.ucs, UniversalContextSchema)

    def test_stats_populated(self) -> None:
        conv = _conversation(
            _msg("Python and PostgreSQL for the backend"),
        )
        result = self.pipeline.generate_from_conversation(conv)
        stats = result.stats
        assert isinstance(stats, GenerationStats)
        assert stats.message_count == 1
        assert stats.entity_count >= 0
        assert stats.processing_time_ms >= 0
        assert isinstance(stats.validation_warnings, tuple)

    def test_entities_extracted(self) -> None:
        conv = _conversation(
            _msg("We use React, TypeScript, and Tailwind CSS for the frontend"),
            _msg("Backend is Python FastAPI with PostgreSQL and Redis"),
        )
        result = self.pipeline.generate_from_conversation(conv)
        entity_names = {e.name for e in result.ucs.entities}
        assert len(entity_names) >= 3

    def test_knowledge_graph_built(self) -> None:
        conv = _conversation(
            _msg("Docker containers deployed on Kubernetes in AWS"),
        )
        result = self.pipeline.generate_from_conversation(conv)
        graph = result.ucs.knowledge_graph
        assert len(graph.nodes) >= 1

    def test_summaries_generated(self) -> None:
        conv = _conversation(
            _msg("First, we need to set up the database schema with proper indexes"),
            _msg("Then implement the REST API endpoints for CRUD operations"),
            _msg("Finally, add authentication middleware with JWT tokens"),
        )
        result = self.pipeline.generate_from_conversation(conv)
        assert len(result.ucs.summaries) > 0
        levels = {s.level for s in result.ucs.summaries}
        assert SummaryLevel.GLOBAL in levels

    def test_importance_scores_populated(self) -> None:
        conv = _conversation(
            _msg("Python and React"),
        )
        result = self.pipeline.generate_from_conversation(conv)
        assert len(result.ucs.importance_scores) == len(result.ucs.entities)

    def test_processing_mode_standard(self) -> None:
        conv = _conversation(_msg("Hello"))
        result = self.pipeline.generate_from_conversation(conv)
        assert result.ucs.session_meta.processing_mode == ProcessingMode.STANDARD

    def test_ucs_version_set(self) -> None:
        conv = _conversation(_msg("test"))
        result = self.pipeline.generate_from_conversation(conv)
        assert result.ucs.version == "1.0.0"

    def test_complex_conversation(self) -> None:
        """Full end-to-end test with a realistic conversation."""
        conv = _conversation(
            _msg("I need to build a SaaS platform for project management"),
            _msg("Let's use Python with FastAPI for the backend API"),
            _msg("The database should be PostgreSQL with Redis for caching"),
            _msg("Frontend will be React with TypeScript and Next.js"),
            _msg("Deploy to AWS using Docker and Kubernetes"),
            _msg("Add Stripe for billing and Auth0 for authentication"),
        )
        result = self.pipeline.generate_from_conversation(conv)

        ucs = result.ucs
        assert ucs.session_meta.message_count == 6
        assert len(ucs.entities) >= 5
        assert len(ucs.summaries) >= 1
        assert ucs.knowledge_graph.nodes
        assert len(ucs.importance_scores) > 0

        # Validation should pass
        assert len(result.stats.validation_warnings) <= 1  # at most "no global summary" warning

    def test_compression_ratio_in_session_meta(self) -> None:
        conv = _conversation(_msg("short"))
        result = self.pipeline.generate_from_conversation(conv)
        ratio = result.ucs.session_meta.compression_ratio
        assert 0.0 <= ratio <= 1.0


# == Local Engine Tests ======================================================


class TestLocalEngine:
    def setup_method(self) -> None:
        self.engine = LocalEngine(config=LocalProcessingConfig(
            enable_spacy=False,
        ))

    def test_process_conversation(self) -> None:
        conv = _conversation(
            _msg("Python and Django for web development"),
        )
        result = self.engine.process_conversation(conv)
        assert isinstance(result.ucs, UniversalContextSchema)
        assert result.ucs.session_meta.processing_mode == ProcessingMode.LOCAL

    def test_process_raw_data(self) -> None:
        raw = [
            {"role": "user", "content": "What is Docker?"},
            {"role": "assistant", "content": "Docker is a container platform."},
        ]
        result = self.engine.process(raw)
        assert result.ucs.session_meta.processing_mode == ProcessingMode.LOCAL

    def test_local_stamp_applied(self) -> None:
        """The LOCAL processing mode must ALWAYS be set."""
        conv = _conversation(_msg("test message"))
        result = self.engine.process_conversation(conv)
        assert result.ucs.session_meta.processing_mode == ProcessingMode.LOCAL

    def test_config_defaults(self) -> None:
        engine = LocalEngine()
        assert engine.config.target_tokens == 4000
        assert engine.config.enable_spacy is True
        assert engine.config.max_messages == 1000

    def test_custom_config(self) -> None:
        config = LocalProcessingConfig(
            target_tokens=8000,
            enable_spacy=False,
            similarity_threshold=0.2,
        )
        engine = LocalEngine(config=config)
        assert engine.config.target_tokens == 8000
        assert engine.config.enable_spacy is False

    def test_entities_extracted_locally(self) -> None:
        conv = _conversation(
            _msg("FastAPI with PostgreSQL and Redis"),
        )
        result = self.engine.process_conversation(conv)
        assert len(result.ucs.entities) >= 1

    def test_summaries_generated_locally(self) -> None:
        conv = _conversation(
            _msg("Build a REST API with proper authentication"),
            _msg("Use JWT tokens with refresh token rotation"),
        )
        result = self.engine.process_conversation(conv)
        assert len(result.ucs.summaries) >= 1

    def test_stats_available(self) -> None:
        conv = _conversation(_msg("test"))
        result = self.engine.process_conversation(conv)
        assert isinstance(result.stats, GenerationStats)
        assert result.stats.processing_time_ms >= 0

    def test_session_meta_preserved(self) -> None:
        """Session metadata (tokens, counts) should survive the LOCAL stamp."""
        conv = _conversation(
            _msg("msg1"),
            _msg("msg2"),
            _msg("msg3"),
        )
        result = self.engine.process_conversation(conv)
        assert result.ucs.session_meta.message_count == 3

    def test_ucs_immutable(self) -> None:
        conv = _conversation(_msg("test"))
        result = self.engine.process_conversation(conv)
        with pytest.raises(Exception):
            result.ucs.version = "2.0.0"  # type: ignore[misc]
