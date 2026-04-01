"""Full UCS generation pipeline orchestrator.

The UCS Generator is the BRAIN'S MAIN LOOP. It takes raw conversation data
(in any format) and produces a complete, validated Universal Context Schema
that can be stored, transferred, and consumed by any target LLM.

Pipeline flow:
    Raw Input (JSON/text)
        -> Parser (auto-detect format, produce CCR)
        -> Entity Extraction (NER + technical + knowledge graph)
        -> Summarization (3-tier hierarchical)
        -> Compression (fit to target token budget)
        -> UCS Assembly (combine all outputs into final schema)
        -> Validation (cross-reference integrity check)
        -> UniversalContextSchema (ready for storage/transfer)

This is the single entry point for all context processing. Everything
else in the engine/ directory is a component that this orchestrator calls.

Usage:
    from app.core.engine.ucs_generator import UCSGeneratorPipeline

    pipeline = UCSGeneratorPipeline()
    ucs = pipeline.generate(raw_data)
    # ucs is a fully validated UniversalContextSchema
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.engine.ccr import Conversation, SourceFormat
from app.core.engine.compressor import CompressionPipeline, CompressionResult
from app.core.engine.entities import EntityPipeline, EntityResult
from app.core.engine.parser import ParserRegistry, create_default_registry
from app.core.engine.summarizer import SummarizationPipeline, SummaryResult
from app.core.models.ucs import (
    Preferences,
    ProcessingMode,
    SessionMeta,
    SourceLLM,
    UniversalContextSchema,
    UCSValidator,
)


# -- Source format to SourceLLM mapping --------------------------------------

_FORMAT_TO_LLM: dict[SourceFormat, SourceLLM] = {
    SourceFormat.OPENAI: SourceLLM.OPENAI,
    SourceFormat.OPENAI_EXPORT: SourceLLM.OPENAI,
    SourceFormat.CLAUDE: SourceLLM.ANTHROPIC,
    SourceFormat.GEMINI: SourceLLM.GOOGLE,
    SourceFormat.GENERIC: SourceLLM.UNKNOWN,
    SourceFormat.UNKNOWN: SourceLLM.UNKNOWN,
}


# -- Pipeline Stats ----------------------------------------------------------


@dataclass(frozen=True)
class GenerationStats:
    """Statistics from the UCS generation pipeline.

    Useful for logging, monitoring, and debugging.
    """

    source_format: str
    source_model: str
    message_count: int
    total_tokens: int
    entity_count: int
    relationship_count: int
    summary_count: int
    topic_count: int
    compression_ratio: float
    spacy_available: bool
    validation_warnings: tuple[str, ...]
    processing_time_ms: float


@dataclass(frozen=True)
class GenerationResult:
    """Complete result from the UCS generation pipeline.

    Attributes:
        ucs: The generated Universal Context Schema.
        conversation: The intermediate CCR Conversation.
        stats: Pipeline statistics.
    """

    ucs: UniversalContextSchema
    conversation: Conversation
    stats: GenerationStats


# -- Main Pipeline -----------------------------------------------------------


class UCSGeneratorPipeline:
    """The brain's main processing pipeline.

    Takes raw conversation data (any format) and produces a complete,
    validated Universal Context Schema.

    This is the orchestrator — it coordinates Parser, Entity Extraction,
    Summarization, and Compression into a single coherent pipeline.

    Usage:
        pipeline = UCSGeneratorPipeline()

        # From raw data (auto-detect format)
        result = pipeline.generate(raw_json_data)

        # From pre-parsed conversation
        result = pipeline.generate_from_conversation(conversation)

        # Access the UCS
        ucs = result.ucs
    """

    def __init__(
        self,
        target_tokens: int = 4000,
        enable_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        similarity_threshold: float = 0.15,
    ) -> None:
        """Initialize the UCS generation pipeline.

        Args:
            target_tokens: Token budget for compression.
            enable_spacy: Whether to use spaCy for NER.
            spacy_model: spaCy model name.
            similarity_threshold: Topic detection similarity threshold.
        """
        self._parser_registry = create_default_registry()
        self._entity_pipeline = EntityPipeline(
            spacy_model=spacy_model,
            enable_spacy=enable_spacy,
        )
        self._summarization_pipeline = SummarizationPipeline(
            similarity_threshold=similarity_threshold,
        )
        self._compression_pipeline = CompressionPipeline(
            target_tokens=target_tokens,
        )

    def generate(self, raw_data: Any) -> GenerationResult:
        """Generate a UCS from raw conversation data.

        Parses the input, extracts entities, summarizes, compresses,
        and assembles the final UCS.

        Args:
            raw_data: Raw conversation data in any supported format
                      (dict, list, or string).

        Returns:
            GenerationResult with UCS, conversation, and stats.
        """
        start_time = datetime.now(timezone.utc)

        # Stage 1: Parse
        conversation = self._parser_registry.parse(raw_data)

        result = self.generate_from_conversation(conversation, start_time)
        return result

    def generate_from_conversation(
        self,
        conversation: Conversation,
        start_time: datetime | None = None,
    ) -> GenerationResult:
        """Generate a UCS from a pre-parsed CCR Conversation.

        Useful when you've already parsed the conversation and want
        to skip the parser stage.

        Args:
            conversation: A normalized CCR Conversation.
            start_time: Optional start time for timing stats.

        Returns:
            GenerationResult with UCS, conversation, and stats.
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        # Stage 2: Entity Extraction
        entity_result = self._entity_pipeline.extract(conversation)

        # Stage 3: Summarization
        summary_result = self._summarization_pipeline.summarize(conversation)

        # Stage 4: Compression
        compression_result = self._compression_pipeline.compress(
            summaries=summary_result.summaries,
            entities=entity_result.entities,
            importance_scores=entity_result.scores,
        )

        # Stage 5: Assemble UCS
        ucs = self._assemble_ucs(
            conversation=conversation,
            entity_result=entity_result,
            summary_result=summary_result,
            compression_result=compression_result,
        )

        # Stage 6: Validate
        warnings = UCSValidator.validate(ucs)

        # Compute timing
        end_time = datetime.now(timezone.utc)
        processing_ms = (end_time - start_time).total_seconds() * 1000

        stats = GenerationStats(
            source_format=conversation.source_format.value,
            source_model=conversation.source_model,
            message_count=conversation.message_count,
            total_tokens=conversation.total_tokens,
            entity_count=len(compression_result.entities),
            relationship_count=entity_result.relationship_count,
            summary_count=len(compression_result.summaries),
            topic_count=len(summary_result.topic_clusters),
            compression_ratio=compression_result.compression_ratio,
            spacy_available=entity_result.spacy_available,
            validation_warnings=tuple(warnings),
            processing_time_ms=round(processing_ms, 2),
        )

        return GenerationResult(
            ucs=ucs,
            conversation=conversation,
            stats=stats,
        )

    def _assemble_ucs(
        self,
        conversation: Conversation,
        entity_result: EntityResult,
        summary_result: SummaryResult,
        compression_result: CompressionResult,
    ) -> UniversalContextSchema:
        """Assemble the final UCS from pipeline outputs.

        Combines compressed entities, summaries, topic clusters,
        knowledge graph, and session metadata into a single
        validated UCS document.
        """
        source_llm = _FORMAT_TO_LLM.get(
            conversation.source_format, SourceLLM.UNKNOWN,
        )

        session_meta = SessionMeta(
            source_llm=source_llm,
            source_model=conversation.source_model,
            total_tokens=conversation.total_tokens,
            message_count=conversation.message_count,
            compression_ratio=compression_result.compression_ratio,
            processing_mode=ProcessingMode.STANDARD,
        )

        return UniversalContextSchema(
            session_meta=session_meta,
            entities=compression_result.entities,
            summaries=compression_result.summaries,
            knowledge_graph=entity_result.graph,
            topic_clusters=summary_result.topic_clusters,
            importance_scores=entity_result.scores,
            preferences=Preferences(),
        )
