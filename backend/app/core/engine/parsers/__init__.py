"""Conversation parsers for all supported LLM formats.

Each parser normalizes a specific LLM format into the Canonical Conversation
Representation (CCR). The ParserRegistry auto-detects the format and routes
to the correct parser.

Supported formats:
    - OpenAI API messages + ChatGPT export JSON
    - Anthropic Claude API messages
    - Google Gemini API content
    - Generic markdown/text (copy-paste fallback)
"""

from app.core.engine.parsers.base import BaseParser, ParserRegistry
from app.core.engine.parsers.claude import ClaudeParser
from app.core.engine.parsers.gemini import GeminiParser
from app.core.engine.parsers.generic import GenericParser
from app.core.engine.parsers.openai import OpenAIParser

__all__ = [
    "BaseParser",
    "ParserRegistry",
    "OpenAIParser",
    "ClaudeParser",
    "GeminiParser",
    "GenericParser",
]
