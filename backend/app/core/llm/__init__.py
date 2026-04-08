"""Unified LLM client layer for Kangaroo Shift.

Provides a single async interface for calling OpenAI, Anthropic, and Google
LLMs — both synchronous (full response) and streaming (chunk-by-chunk).

Usage:
    from app.core.llm.client import LLMClient

    client = LLMClient()
    response = await client.call("openai", "gpt-4o-mini", messages)

    async for chunk in client.call_streaming("anthropic", "claude-sonnet-4", messages):
        print(chunk.delta, end="")
"""

from app.core.llm.models import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMResponse,
    ModelInfo,
    ModelProvider,
    MODEL_REGISTRY,
)

__all__ = [
    "LLMChunk",
    "LLMError",
    "LLMMessage",
    "LLMResponse",
    "ModelInfo",
    "ModelProvider",
    "MODEL_REGISTRY",
]
