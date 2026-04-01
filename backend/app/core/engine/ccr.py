"""Canonical Conversation Representation (CCR).

The CCR is the brain's internal language. Every conversation from every LLM
is normalized into this format before any processing happens. It doesn't
matter if the user pasted a ChatGPT JSON export, a Claude API response,
a Gemini multi-part message, or raw markdown — the CCR looks identical.

Why a separate format from UCS?
    - CCR is the INPUT to the brain (raw normalized conversation)
    - UCS is the OUTPUT of the brain (processed, enriched context)
    - CCR preserves the full conversation (every message, verbatim)
    - UCS is compressed and augmented (summaries, entities, knowledge graph)

Design principles:
    1. Immutable — frozen Pydantic models, no mutation
    2. Lossless — nothing from the source format is discarded
    3. Multi-modal — text, code, images, files, tool calls all represented
    4. Ordered — messages maintain their original sequence
    5. Metadata-rich — source-specific metadata preserved in extra fields

Usage:
    from app.core.engine.ccr import Conversation, Message, ContentBlock

    # Parser produces a Conversation
    conversation = openai_parser.parse(raw_data)

    # Downstream components consume it
    entities = entity_extractor.extract(conversation)
    summaries = summarizer.summarize(conversation)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# -- Enums -------------------------------------------------------------------


class MessageRole(str, Enum):
    """Roles in a conversation. Every LLM uses some variant of these."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    """Types of content within a message.

    A single message can contain multiple content blocks of different types.
    Example: an assistant message might contain text + code + image.
    """

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    ERROR = "error"


class SourceFormat(str, Enum):
    """Identifies which parser produced this CCR."""

    OPENAI = "openai"
    OPENAI_EXPORT = "openai_export"
    CLAUDE = "claude"
    GEMINI = "gemini"
    GENERIC = "generic"
    UNKNOWN = "unknown"


# -- Content Blocks ----------------------------------------------------------


class ContentBlock(BaseModel):
    """A single piece of content within a message.

    Messages are not just text — they can contain code blocks, images, file
    references, tool calls, and tool results. Each of these is a ContentBlock.

    This is the atomic unit of conversation content. The parser breaks down
    complex multi-modal messages into a sequence of ContentBlocks.

    Examples:
        - Text: ContentBlock(type="text", text="Hello, how are you?")
        - Code: ContentBlock(type="code", text="def foo(): ...", language="python")
        - Image: ContentBlock(type="image", url="https://...", alt_text="diagram")
        - Tool call: ContentBlock(type="tool_call", text='{"name":"search",...}',
                                   tool_name="search", tool_call_id="call_123")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ContentType
    text: str = Field(
        default="",
        description="Primary text content. For code blocks, this is the code.",
    )
    language: str = Field(
        default="",
        description="Programming language (for code blocks). Empty for non-code.",
    )
    url: str = Field(
        default="",
        description="URL for images, files, or external references.",
    )
    alt_text: str = Field(
        default="",
        description="Alt text for images. Empty for non-image blocks.",
    )
    tool_name: str = Field(
        default="",
        description="Tool/function name (for tool_call and tool_result blocks).",
    )
    tool_call_id: str = Field(
        default="",
        description="Unique identifier linking a tool_call to its tool_result.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata preserved from the original format.",
    )

    @property
    def is_empty(self) -> bool:
        """Check if this content block has no meaningful content."""
        return not self.text.strip() and not self.url

    @property
    def text_content(self) -> str:
        """Get the text representation of this block for downstream processing.

        For code blocks, wraps in markdown fences.
        For tool calls, returns the tool name and arguments.
        For images, returns a reference string.
        """
        if self.type == ContentType.CODE and self.language:
            return f"```{self.language}\n{self.text}\n```"
        if self.type == ContentType.CODE:
            return f"```\n{self.text}\n```"
        if self.type == ContentType.IMAGE:
            return f"[Image: {self.alt_text or self.url}]"
        if self.type == ContentType.TOOL_CALL:
            return f"[Tool call: {self.tool_name}({self.text})]"
        if self.type == ContentType.TOOL_RESULT:
            return f"[Tool result ({self.tool_name}): {self.text}]"
        if self.type == ContentType.THINKING:
            return ""  # Thinking blocks are internal, not shown
        return self.text


# -- Message -----------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation.

    This represents one turn in the conversation — a user message, an
    assistant response, a system instruction, or a tool interaction.

    Messages contain one or more ContentBlocks. A simple text message has
    one TEXT block. A complex assistant response might have TEXT + CODE +
    IMAGE blocks.

    The message preserves all source metadata (model used, token count,
    timestamp, finish reason) for analytics and billing purposes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: tuple[ContentBlock, ...] = Field(
        default=(),
        description="Ordered sequence of content blocks in this message.",
    )
    name: str = Field(
        default="",
        description="Speaker name (for multi-user conversations).",
    )
    model: str = Field(
        default="",
        description="Model that generated this message (for assistant messages).",
    )
    timestamp: datetime | None = Field(
        default=None,
        description="When this message was created. None if not available.",
    )
    token_count: int | None = Field(
        default=None,
        ge=0,
        description="Token count for this message. None if not available.",
    )
    finish_reason: str = Field(
        default="",
        description="Why generation stopped (stop, length, tool_calls, etc.).",
    )
    message_id: str = Field(
        default="",
        description="Original message ID from the source platform.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata preserved from the original format.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime | None:
        """Ensure timestamps are UTC."""
        if v is None:
            return None
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def full_text(self) -> str:
        """Get the full text content of this message.

        Concatenates all content blocks into a single string.
        Used by downstream components (entity extractor, summarizer).
        """
        parts = [block.text_content for block in self.content if block.text_content]
        return "\n\n".join(parts)

    @property
    def has_tool_calls(self) -> bool:
        """Check if this message contains tool/function calls."""
        return any(b.type == ContentType.TOOL_CALL for b in self.content)

    @property
    def has_code(self) -> bool:
        """Check if this message contains code blocks."""
        return any(b.type == ContentType.CODE for b in self.content)

    @property
    def has_images(self) -> bool:
        """Check if this message contains images."""
        return any(b.type == ContentType.IMAGE for b in self.content)

    @property
    def is_empty(self) -> bool:
        """Check if this message has no meaningful content."""
        return all(b.is_empty for b in self.content)


# -- Conversation ------------------------------------------------------------


class Conversation(BaseModel):
    """A complete, normalized conversation from any LLM.

    This is the output of every parser and the input to every downstream
    component in the Context Engine pipeline. No matter what format the
    original conversation was in, it comes out as a Conversation.

    The Conversation:
    - Preserves ALL information from the source format
    - Normalizes message structure across LLMs
    - Separates system instructions from regular messages
    - Computes aggregate statistics (total tokens, message count)
    - Carries source format metadata for debugging and analytics

    Example:
        conv = openai_parser.parse(chatgpt_export_json)
        assert conv.source_format == SourceFormat.OPENAI_EXPORT
        assert conv.messages[0].role == MessageRole.USER
        assert conv.total_tokens > 0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Identity
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique ID for this conversation instance.",
    )

    # Source tracking
    source_format: SourceFormat = SourceFormat.UNKNOWN
    source_llm: str = Field(
        default="unknown",
        description="LLM provider (openai, anthropic, google, etc.).",
    )
    source_model: str = Field(
        default="",
        description="Specific model (gpt-4o, claude-sonnet-4-20250514, gemini-1.5-pro, etc.).",
    )

    # Content
    system_instruction: str = Field(
        default="",
        description="System prompt / instruction, extracted from messages.",
    )
    messages: tuple[Message, ...] = Field(
        default=(),
        description="Ordered sequence of conversation messages (excluding system).",
    )

    # Title / label
    title: str = Field(
        default="",
        description="Conversation title (if available from source).",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Aggregates (computed from messages)
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total token count across all messages.",
    )
    message_count: int = Field(
        default=0,
        ge=0,
        description="Number of non-system messages.",
    )

    # Source metadata
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original metadata from source format, preserved verbatim.",
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamps are UTC."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def user_messages(self) -> tuple[Message, ...]:
        """Get only user messages."""
        return tuple(m for m in self.messages if m.role == MessageRole.USER)

    @property
    def assistant_messages(self) -> tuple[Message, ...]:
        """Get only assistant messages."""
        return tuple(m for m in self.messages if m.role == MessageRole.ASSISTANT)

    @property
    def has_tool_usage(self) -> bool:
        """Check if any message contains tool calls."""
        return any(m.has_tool_calls for m in self.messages)

    @property
    def has_code(self) -> bool:
        """Check if any message contains code."""
        return any(m.has_code for m in self.messages)

    @property
    def all_text(self) -> str:
        """Get all text content from the conversation.

        Used for full-text operations like keyword extraction.
        """
        parts = []
        if self.system_instruction:
            parts.append(self.system_instruction)
        parts.extend(m.full_text for m in self.messages if m.full_text)
        return "\n\n".join(parts)

    @property
    def languages_used(self) -> frozenset[str]:
        """Get all programming languages used in code blocks."""
        languages: set[str] = set()
        for msg in self.messages:
            for block in msg.content:
                if block.type == ContentType.CODE and block.language:
                    languages.add(block.language.lower())
        return frozenset(languages)
