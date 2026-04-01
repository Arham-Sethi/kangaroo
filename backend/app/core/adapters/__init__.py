"""Output adapters -- reconstruct UCS into LLM-specific formats.

Adapters are the OUTPUT side of the pipeline:
    Parser (IN): LLM format -> CCR -> UCS
    Adapter (OUT): UCS -> LLM format

Available adapters:
    - OpenAIAdapter: UCS -> OpenAI Chat Completions format
    - ClaudeAdapter: UCS -> Anthropic Claude Messages format
    - GeminiAdapter: UCS -> Google Gemini generateContent format
"""

from app.core.adapters.base import (
    AdaptedOutput,
    AdapterRegistry,
    BaseAdapter,
    create_default_adapter_registry,
)
from app.core.adapters.claude_adapter import ClaudeAdapter
from app.core.adapters.gemini_adapter import GeminiAdapter
from app.core.adapters.openai_adapter import OpenAIAdapter

__all__ = [
    "AdaptedOutput",
    "AdapterRegistry",
    "BaseAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "create_default_adapter_registry",
]
