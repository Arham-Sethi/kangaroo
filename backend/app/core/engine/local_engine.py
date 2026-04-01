"""Local-only processing mode -- no network calls, extractive summarization.

The Local Processing Engine is the privacy-first mode of Kangaroo Shift.
When enabled, ALL processing happens on the user's machine -- no data
leaves the device, no API calls are made, no telemetry is sent.

This mode uses:
    - Local parsers (all 4 formats, no network)
    - Local entity extraction (spaCy + regex, no API)
    - Extractive summarization (TF-IDF, no LLM calls)
    - Local compression (priority queue, no API)
    - Local UCS assembly and validation

The tradeoff: extractive summaries are lower quality than LLM-generated
abstractive summaries. But for privacy-conscious users, this is
non-negotiable. Their data never leaves their machine.

Usage:
    from app.core.engine.local_engine import LocalEngine

    engine = LocalEngine()
    result = engine.process(raw_data)
    # result.ucs is a fully local UCS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.engine.ccr import Conversation
from app.core.engine.ucs_generator import (
    GenerationResult,
    GenerationStats,
    UCSGeneratorPipeline,
)
from app.core.models.ucs import ProcessingMode, SessionMeta, UniversalContextSchema


@dataclass(frozen=True)
class LocalProcessingConfig:
    """Configuration for local-only processing.

    All fields have sensible defaults for privacy-first operation.
    """

    target_tokens: int = 4000
    enable_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    similarity_threshold: float = 0.15
    max_message_length: int = 50000
    max_messages: int = 1000


class LocalEngine:
    """Privacy-first local processing engine.

    Wraps UCSGeneratorPipeline with local-only guarantees:
        - No network calls (spaCy model must be pre-installed)
        - No API keys required
        - No telemetry
        - All processing on-device
        - Output marked as ProcessingMode.LOCAL

    Usage:
        engine = LocalEngine()
        result = engine.process(raw_data)
        assert result.ucs.session_meta.processing_mode == ProcessingMode.LOCAL
    """

    def __init__(self, config: LocalProcessingConfig | None = None) -> None:
        """Initialize the local engine.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self._config = config or LocalProcessingConfig()
        self._pipeline = UCSGeneratorPipeline(
            target_tokens=self._config.target_tokens,
            enable_spacy=self._config.enable_spacy,
            spacy_model=self._config.spacy_model,
            similarity_threshold=self._config.similarity_threshold,
        )

    @property
    def config(self) -> LocalProcessingConfig:
        """Get the current configuration."""
        return self._config

    def process(self, raw_data: Any) -> GenerationResult:
        """Process raw conversation data locally.

        No network calls. Everything runs on-device.

        Args:
            raw_data: Raw conversation data in any supported format.

        Returns:
            GenerationResult with LOCAL processing mode set.
        """
        result = self._pipeline.generate(raw_data)

        # Override processing mode to LOCAL
        local_ucs = self._stamp_local(result.ucs)

        return GenerationResult(
            ucs=local_ucs,
            conversation=result.conversation,
            stats=result.stats,
        )

    def process_conversation(self, conversation: Conversation) -> GenerationResult:
        """Process a pre-parsed conversation locally.

        Args:
            conversation: A normalized CCR Conversation.

        Returns:
            GenerationResult with LOCAL processing mode set.
        """
        result = self._pipeline.generate_from_conversation(conversation)
        local_ucs = self._stamp_local(result.ucs)

        return GenerationResult(
            ucs=local_ucs,
            conversation=result.conversation,
            stats=result.stats,
        )

    @staticmethod
    def _stamp_local(ucs: UniversalContextSchema) -> UniversalContextSchema:
        """Replace processing mode with LOCAL.

        Creates a new immutable UCS with the processing mode changed.
        """
        new_meta = SessionMeta(
            session_id=ucs.session_meta.session_id,
            created_at=ucs.session_meta.created_at,
            updated_at=ucs.session_meta.updated_at,
            source_llm=ucs.session_meta.source_llm,
            source_model=ucs.session_meta.source_model,
            total_tokens=ucs.session_meta.total_tokens,
            message_count=ucs.session_meta.message_count,
            compression_ratio=ucs.session_meta.compression_ratio,
            processing_mode=ProcessingMode.LOCAL,
        )

        return UniversalContextSchema(
            version=ucs.version,
            session_meta=new_meta,
            entities=ucs.entities,
            summaries=ucs.summaries,
            decisions=ucs.decisions,
            tasks=ucs.tasks,
            preferences=ucs.preferences,
            artifacts=ucs.artifacts,
            knowledge_graph=ucs.knowledge_graph,
            topic_clusters=ucs.topic_clusters,
            safety_flags=ucs.safety_flags,
            llm_comparisons=ucs.llm_comparisons,
            importance_scores=ucs.importance_scores,
        )
