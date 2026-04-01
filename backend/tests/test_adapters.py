"""Tests for output adapters (UCS -> LLM format).

Tests cover:
    1. Base adapter registry
    2. OpenAI adapter output format
    3. Claude adapter output format
    4. Gemini adapter output format
    5. Context summary generation
    6. Cross-adapter consistency

Total: 40+ tests
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.adapters import (
    AdaptedOutput,
    AdapterRegistry,
    BaseAdapter,
    ClaudeAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    create_default_adapter_registry,
)
from app.core.models.ucs import (
    Artifact,
    ArtifactType,
    Decision,
    DecisionStatus,
    Entity,
    EntityType,
    KnowledgeGraph,
    Preferences,
    ProcessingMode,
    SessionMeta,
    SourceLLM,
    Summary,
    SummaryLevel,
    Task,
    TaskStatus,
    UniversalContextSchema,
)


# -- Helpers -----------------------------------------------------------------


def _minimal_ucs() -> UniversalContextSchema:
    """Create a minimal valid UCS for testing."""
    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=5,
            total_tokens=1500,
        ),
    )


def _rich_ucs() -> UniversalContextSchema:
    """Create a feature-rich UCS for thorough testing."""
    entity1_id = uuid4()
    entity2_id = uuid4()

    return UniversalContextSchema(
        session_meta=SessionMeta(
            source_llm=SourceLLM.OPENAI,
            source_model="gpt-4o",
            message_count=10,
            total_tokens=5000,
            compression_ratio=0.6,
        ),
        entities=(
            Entity(
                id=entity1_id,
                name="Python",
                type=EntityType.TECHNOLOGY,
                importance=0.9,
                aliases=("python3",),
            ),
            Entity(
                id=entity2_id,
                name="FastAPI",
                type=EntityType.TECHNOLOGY,
                importance=0.8,
            ),
        ),
        summaries=(
            Summary(
                level=SummaryLevel.GLOBAL,
                content="Building a web API with Python and FastAPI.",
                token_count=10,
                covers_messages=(0, 9),
            ),
            Summary(
                level=SummaryLevel.TOPIC,
                content="Discussion about backend architecture.",
                token_count=6,
                covers_messages=(0, 4),
            ),
        ),
        decisions=(
            Decision(
                description="Use PostgreSQL for the database",
                rationale="Best for relational data with JSON support",
                status=DecisionStatus.ACTIVE,
            ),
        ),
        tasks=(
            Task(
                description="Implement user authentication",
                status=TaskStatus.ACTIVE,
            ),
            Task(
                description="Set up CI/CD pipeline",
                status=TaskStatus.COMPLETED,
            ),
        ),
        artifacts=(
            Artifact(
                type=ArtifactType.CODE,
                language="python",
                content="from fastapi import FastAPI\napp = FastAPI()",
                title="main.py",
            ),
        ),
        preferences=Preferences(domain_expertise=("backend", "python")),
        importance_scores={
            str(entity1_id): 0.9,
            str(entity2_id): 0.8,
        },
    )


# == Registry Tests ==========================================================


class TestAdapterRegistry:
    def test_create_default_registry(self) -> None:
        registry = create_default_adapter_registry()
        assert "openai" in registry.available_formats
        assert "claude" in registry.available_formats
        assert "gemini" in registry.available_formats

    def test_available_formats(self) -> None:
        registry = AdapterRegistry()
        assert registry.available_formats == []

    def test_register_adapter(self) -> None:
        registry = AdapterRegistry()
        registry.register(OpenAIAdapter())
        assert "openai" in registry.available_formats

    def test_get_adapter(self) -> None:
        registry = create_default_adapter_registry()
        adapter = registry.get_adapter("openai")
        assert adapter is not None
        assert adapter.format_name == "openai"

    def test_get_unknown_adapter(self) -> None:
        registry = create_default_adapter_registry()
        assert registry.get_adapter("nonexistent") is None

    def test_adapt_unknown_format(self) -> None:
        registry = create_default_adapter_registry()
        with pytest.raises(ValueError, match="Unknown target format"):
            registry.adapt(_minimal_ucs(), target="nonexistent")

    def test_adapt_openai(self) -> None:
        registry = create_default_adapter_registry()
        output = registry.adapt(_minimal_ucs(), target="openai")
        assert isinstance(output, AdaptedOutput)
        assert output.format_name == "openai"


# == OpenAI Adapter Tests ====================================================


class TestOpenAIAdapter:
    def setup_method(self) -> None:
        self.adapter = OpenAIAdapter()

    def test_format_name(self) -> None:
        assert self.adapter.format_name == "openai"

    def test_minimal_ucs(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert output.format_name == "openai"
        assert len(output.messages) >= 2  # system + continuation
        assert output.messages[0]["role"] == "system"
        assert output.token_estimate > 0

    def test_rich_ucs(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert len(output.messages) >= 2
        # System prompt should contain context
        system_content = output.messages[0]["content"]
        assert "Python" in system_content or "FastAPI" in system_content

    def test_system_prompt_contains_summary(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "Previous Conversation Summary" in output.system_prompt

    def test_system_prompt_contains_entities(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "Python" in output.system_prompt

    def test_system_prompt_contains_decisions(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "PostgreSQL" in output.system_prompt

    def test_system_prompt_contains_tasks(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "authentication" in output.system_prompt

    def test_artifacts_included(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        # Artifacts show as assistant messages
        all_content = " ".join(m.get("content", "") for m in output.messages)
        assert "FastAPI" in all_content

    def test_model_suggestion(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert "model_suggestion" in output.metadata

    def test_metadata_populated(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert output.metadata["source_llm"] == "openai"
        assert output.metadata["entity_count"] == 2


# == Claude Adapter Tests ====================================================


class TestClaudeAdapter:
    def setup_method(self) -> None:
        self.adapter = ClaudeAdapter()

    def test_format_name(self) -> None:
        assert self.adapter.format_name == "claude"

    def test_minimal_ucs(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert output.format_name == "claude"
        assert len(output.messages) >= 2

    def test_messages_start_with_user(self) -> None:
        """Claude requires first message to be user role."""
        output = self.adapter.adapt(_rich_ucs())
        assert output.messages[0]["role"] == "user"

    def test_messages_alternate(self) -> None:
        """Claude requires alternating user/assistant."""
        output = self.adapter.adapt(_rich_ucs())
        roles = [m["role"] for m in output.messages]
        for i in range(1, len(roles)):
            assert roles[i] != roles[i - 1]

    def test_content_is_typed_blocks(self) -> None:
        """Claude uses typed content blocks [{type: 'text', text: '...'}]."""
        output = self.adapter.adapt(_rich_ucs())
        first_msg = output.messages[0]
        assert isinstance(first_msg["content"], list)
        assert first_msg["content"][0]["type"] == "text"

    def test_system_prompt_separate(self) -> None:
        """System prompt should be separate (not a message)."""
        output = self.adapter.adapt(_rich_ucs())
        assert len(output.system_prompt) > 0
        # No message should have role "system"
        for msg in output.messages:
            assert msg["role"] != "system"

    def test_rich_ucs_context(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "Python" in output.system_prompt
        assert output.metadata["api_format"] == "messages"

    def test_model_suggestion(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert "model_suggestion" in output.metadata


# == Gemini Adapter Tests ====================================================


class TestGeminiAdapter:
    def setup_method(self) -> None:
        self.adapter = GeminiAdapter()

    def test_format_name(self) -> None:
        assert self.adapter.format_name == "gemini"

    def test_minimal_ucs(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert output.format_name == "gemini"
        assert len(output.messages) >= 2

    def test_uses_model_role(self) -> None:
        """Gemini uses 'model' not 'assistant'."""
        output = self.adapter.adapt(_rich_ucs())
        roles = {m["role"] for m in output.messages}
        assert "model" in roles
        assert "assistant" not in roles

    def test_uses_parts_format(self) -> None:
        """Gemini uses 'parts' not 'content'."""
        output = self.adapter.adapt(_rich_ucs())
        for msg in output.messages:
            assert "parts" in msg
            assert isinstance(msg["parts"], list)

    def test_system_instruction_in_metadata(self) -> None:
        """Gemini system instruction should be in metadata."""
        output = self.adapter.adapt(_rich_ucs())
        assert "system_instruction" in output.metadata
        assert "parts" in output.metadata["system_instruction"]

    def test_rich_ucs_context(self) -> None:
        output = self.adapter.adapt(_rich_ucs())
        assert "Python" in output.system_prompt

    def test_model_suggestion(self) -> None:
        output = self.adapter.adapt(_minimal_ucs())
        assert "model_suggestion" in output.metadata


# == Cross-Adapter Consistency Tests =========================================


class TestCrossAdapterConsistency:
    """Verify all adapters produce consistent outputs from the same UCS."""

    def setup_method(self) -> None:
        self.registry = create_default_adapter_registry()
        self.ucs = _rich_ucs()

    def test_all_formats_produce_output(self) -> None:
        for fmt in self.registry.available_formats:
            output = self.registry.adapt(self.ucs, target=fmt)
            assert isinstance(output, AdaptedOutput)
            assert output.format_name == fmt
            assert len(output.messages) >= 2
            assert output.token_estimate > 0

    def test_all_formats_have_system_prompt(self) -> None:
        for fmt in self.registry.available_formats:
            output = self.registry.adapt(self.ucs, target=fmt)
            assert len(output.system_prompt) > 0

    def test_all_formats_mention_entities(self) -> None:
        for fmt in self.registry.available_formats:
            output = self.registry.adapt(self.ucs, target=fmt)
            assert "Python" in output.system_prompt

    def test_all_formats_mention_decisions(self) -> None:
        for fmt in self.registry.available_formats:
            output = self.registry.adapt(self.ucs, target=fmt)
            assert "PostgreSQL" in output.system_prompt

    def test_all_formats_have_metadata(self) -> None:
        for fmt in self.registry.available_formats:
            output = self.registry.adapt(self.ucs, target=fmt)
            assert "model_suggestion" in output.metadata
            assert "source_llm" in output.metadata
