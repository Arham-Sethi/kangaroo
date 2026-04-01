"""Base parser interface and format auto-detection registry.

The ParserRegistry is the entry point for all conversation parsing. It:
    1. Accepts raw data (dict, list, or string)
    2. Auto-detects the source format by examining structural markers
    3. Routes to the correct parser
    4. Returns a normalized Conversation (CCR)

Detection strategy:
    - OpenAI API: list of dicts with 'role' + 'content' keys
    - ChatGPT export: dict with 'mapping' key (ChatGPT's conversation tree)
    - Claude API: dict with 'content' as list of typed blocks
    - Gemini API: dict with 'candidates' or list with 'parts' key
    - Generic: string input (markdown, plain text, etc.)

Each parser is registered with a priority. When auto-detecting, we try
higher-priority parsers first. If a parser's `can_parse()` returns True,
we use it. This allows graceful fallback to the generic parser.
"""

from __future__ import annotations

import json
import structlog
from abc import ABC, abstractmethod
from typing import Any

from app.core.engine.ccr import Conversation, SourceFormat

logger = structlog.get_logger()


class ParseError(Exception):
    """Raised when a parser cannot process the input data.

    Contains enough context for debugging: which parser, what went wrong,
    and optionally a snippet of the problematic data.
    """

    def __init__(
        self,
        message: str,
        parser_name: str = "",
        source_format: SourceFormat = SourceFormat.UNKNOWN,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.parser_name = parser_name
        self.source_format = source_format
        self.details = details or {}
        super().__init__(message)


class BaseParser(ABC):
    """Abstract base class for all conversation parsers.

    Every format-specific parser inherits from this and implements:
        - can_parse(): Check if raw data matches this format
        - parse(): Convert raw data to a Conversation

    Parsers are stateless — they don't maintain any state between calls.
    This makes them safe to use concurrently and easy to test.
    """

    # Higher priority = tried first during auto-detection.
    # Specific formats (OpenAI, Claude) should have higher priority
    # than the generic fallback parser.
    priority: int = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable parser name for logging and errors."""
        ...

    @property
    @abstractmethod
    def source_format(self) -> SourceFormat:
        """The source format this parser handles."""
        ...

    @abstractmethod
    def can_parse(self, data: Any) -> bool:
        """Check if this parser can handle the given data.

        Must be fast (no deep parsing) — just check structural markers.
        Should never raise exceptions; return False on any error.

        Args:
            data: Raw input data (dict, list, or string).

        Returns:
            True if this parser can handle the data.
        """
        ...

    @abstractmethod
    def parse(self, data: Any) -> Conversation:
        """Parse raw data into a normalized Conversation.

        Args:
            data: Raw input data in this parser's expected format.

        Returns:
            A normalized Conversation (CCR).

        Raises:
            ParseError: If the data cannot be parsed.
        """
        ...

    def _safe_get(self, data: dict, key: str, default: Any = None) -> Any:
        """Safely get a value from a dict, returning default if missing."""
        if not isinstance(data, dict):
            return default
        return data.get(key, default)

    def _safe_str(self, value: Any) -> str:
        """Convert any value to a string safely."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Convert any value to an int safely."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default


class ParserRegistry:
    """Registry of all conversation parsers with auto-detection.

    The registry maintains a list of parser instances sorted by priority.
    When `parse()` is called, it tries each parser's `can_parse()` method
    in priority order and uses the first match.

    This is the ONLY entry point for conversation parsing in the entire
    application. All routes, CLI commands, and SDK methods call this.

    Usage:
        registry = ParserRegistry()
        conversation = registry.parse(raw_data)
        # conversation is a normalized CCR regardless of input format
    """

    def __init__(self) -> None:
        self._parsers: list[BaseParser] = []

    def register(self, parser: BaseParser) -> None:
        """Register a parser and maintain priority order."""
        self._parsers.append(parser)
        self._parsers.sort(key=lambda p: p.priority, reverse=True)

    @property
    def parsers(self) -> tuple[BaseParser, ...]:
        """Get all registered parsers in priority order."""
        return tuple(self._parsers)

    def detect_format(self, data: Any) -> BaseParser | None:
        """Detect the format of raw data and return the matching parser.

        Tries each parser in priority order. Returns None if no parser
        can handle the data (should rarely happen since GenericParser
        handles strings).
        """
        for parser in self._parsers:
            try:
                if parser.can_parse(data):
                    return parser
            except Exception:
                # can_parse should never raise, but be defensive
                continue
        return None

    def parse(self, data: Any) -> Conversation:
        """Parse raw data into a normalized Conversation.

        This is the main entry point. It:
            1. Normalizes input (JSON string → dict/list)
            2. Auto-detects format
            3. Routes to the correct parser
            4. Returns a Conversation

        Args:
            data: Raw input — can be dict, list, string (text or JSON).

        Returns:
            A normalized Conversation (CCR).

        Raises:
            ParseError: If no parser can handle the data or parsing fails.
        """
        # Step 1: If it's a JSON string, try to parse it
        parsed_data = self._normalize_input(data)

        # Step 2: Auto-detect format
        parser = self.detect_format(parsed_data)
        if parser is None:
            raise ParseError(
                "No parser could handle the input data. "
                "Supported formats: OpenAI, Claude, Gemini, markdown/text.",
                parser_name="registry",
            )

        # Step 3: Parse
        try:
            conversation = parser.parse(parsed_data)
            logger.info(
                "conversation_parsed",
                parser=parser.name,
                format=parser.source_format.value,
                messages=conversation.message_count,
                tokens=conversation.total_tokens,
            )
            return conversation
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                f"Parser '{parser.name}' failed: {exc}",
                parser_name=parser.name,
                source_format=parser.source_format,
            ) from exc

    def _normalize_input(self, data: Any) -> Any:
        """Normalize input data — parse JSON strings to Python objects.

        If the input is a string that looks like JSON, parse it.
        Otherwise, return as-is (string for generic parser, dict/list
        for format-specific parsers).
        """
        if isinstance(data, str):
            stripped = data.strip()
            if stripped and stripped[0] in ("{", "["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass  # Not valid JSON — treat as plain text
        return data


def create_default_registry() -> ParserRegistry:
    """Create a ParserRegistry with all built-in parsers registered.

    This is the factory function used by the application. It registers
    parsers in the correct priority order:
        1. OpenAI (priority 100) — most common format
        2. Claude (priority 90)
        3. Gemini (priority 80)
        4. Generic (priority 0) — fallback
    """
    from app.core.engine.parsers.claude import ClaudeParser
    from app.core.engine.parsers.gemini import GeminiParser
    from app.core.engine.parsers.generic import GenericParser
    from app.core.engine.parsers.openai import OpenAIParser

    registry = ParserRegistry()
    registry.register(OpenAIParser())
    registry.register(ClaudeParser())
    registry.register(GeminiParser())
    registry.register(GenericParser())
    return registry
