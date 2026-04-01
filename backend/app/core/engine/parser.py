"""Conversation parsers — public API.

Re-exports from the parsers subpackage for backward compatibility
and convenience. Import from here or from parsers/ directly.

Usage:
    from app.core.engine.parser import create_default_registry

    registry = create_default_registry()
    conversation = registry.parse(raw_data)
"""

from app.core.engine.parsers.base import (
    BaseParser,
    ParseError,
    ParserRegistry,
    create_default_registry,
)
from app.core.engine.parsers.claude import ClaudeParser
from app.core.engine.parsers.gemini import GeminiParser
from app.core.engine.parsers.generic import GenericParser
from app.core.engine.parsers.openai import OpenAIParser

__all__ = [
    "BaseParser",
    "ParseError",
    "ParserRegistry",
    "create_default_registry",
    "OpenAIParser",
    "ClaudeParser",
    "GeminiParser",
    "GenericParser",
]
