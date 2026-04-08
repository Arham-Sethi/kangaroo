"""Unified async LLM client for OpenAI, Anthropic, and Google APIs.

This is the single integration point for all LLM calls in Kangaroo Shift.
The Cockpit orchestrator, Workflow engine, Chain executor, and Consensus
engine all use this client — no other module should call LLM APIs directly.

Design:
    - Each provider has its own private method (_call_openai, _call_anthropic, etc.)
    - The public `call()` method routes to the correct provider based on model ID
    - Streaming uses `call_streaming()` which yields LLMChunk objects
    - All errors are wrapped into LLMError for uniform handling
    - httpx is used directly (no SDK dependencies) for maximum control

Usage:
    client = LLMClient()
    response = await client.call("gpt-4o-mini", [LLMMessage("user", "Hello")])

    async for chunk in client.call_streaming("claude-sonnet-4", messages):
        print(chunk.delta, end="")
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.core.llm.models import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMResponse,
    ModelInfo,
    ModelProvider,
    resolve_model,
)

logger = structlog.get_logger()

# Default timeout: 60s connect, 120s read (streaming can be slow)
_DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class LLMClient:
    """Async LLM client supporting OpenAI, Anthropic, and Google.

    Thread-safe: each call creates its own httpx client. No shared
    mutable state. Suitable for concurrent use in asyncio.gather().
    """

    def __init__(self, timeout: httpx.Timeout | None = None) -> None:
        self._timeout = timeout or _DEFAULT_TIMEOUT
        self._settings = get_settings()

    # ── Public API ───────────────────────────────────────────────────────

    async def call(
        self,
        model_id: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Call an LLM and return the complete response.

        Args:
            model_id: Model identifier (e.g., "gpt-4o", "claude-sonnet-4").
            messages: Conversation messages.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with content, token counts, cost, and latency.

        Raises:
            LLMError: On any API failure (timeout, auth, rate limit, etc.).
        """
        model_info = resolve_model(model_id)
        self._validate_api_key(model_info.provider)

        start = time.monotonic()

        if model_info.provider == ModelProvider.OPENAI:
            response = await self._call_openai(model_info, messages, temperature, max_tokens)
        elif model_info.provider == ModelProvider.ANTHROPIC:
            response = await self._call_anthropic(model_info, messages, temperature, max_tokens)
        elif model_info.provider == ModelProvider.GOOGLE:
            response = await self._call_google(model_info, messages, temperature, max_tokens)
        else:
            raise LLMError(f"Unsupported provider: {model_info.provider}", provider=model_info.provider)

        latency_ms = (time.monotonic() - start) * 1000
        cost = model_info.estimate_cost(response.prompt_tokens, response.completion_tokens)

        logger.info(
            "llm_call_complete",
            model=model_id,
            provider=model_info.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=cost,
        )

        return LLMResponse(
            content=response.content,
            model=response.model,
            provider=model_info.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=cost,
            metadata=response.metadata,
        )

    async def call_streaming(
        self,
        model_id: str,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Stream an LLM response chunk by chunk.

        The final chunk has is_final=True and includes token counts.

        Args:
            model_id: Model identifier.
            messages: Conversation messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            LLMChunk objects with delta text.

        Raises:
            LLMError: On any API failure.
        """
        model_info = resolve_model(model_id)
        self._validate_api_key(model_info.provider)

        if model_info.provider == ModelProvider.OPENAI:
            async for chunk in self._stream_openai(model_info, messages, temperature, max_tokens):
                yield chunk
        elif model_info.provider == ModelProvider.ANTHROPIC:
            async for chunk in self._stream_anthropic(model_info, messages, temperature, max_tokens):
                yield chunk
        elif model_info.provider == ModelProvider.GOOGLE:
            async for chunk in self._stream_google(model_info, messages, temperature, max_tokens):
                yield chunk
        else:
            raise LLMError(f"Unsupported provider: {model_info.provider}", provider=model_info.provider)

    # ── OpenAI ───────────────────────────────────────────────────────────

    async def _call_openai(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call OpenAI Chat Completions API."""
        api_key = self._get_key(ModelProvider.OPENAI)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload = {
            "model": model.api_model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        data = await self._post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            payload=payload,
            provider="openai",
            model_id=model.model_id,
        )

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", model.api_model_name),
            provider=ModelProvider.OPENAI,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=0.0,  # Overridden by caller
            cost_usd=0.0,  # Overridden by caller
            metadata={"finish_reason": choice.get("finish_reason")},
        )

    async def _stream_openai(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[LLMChunk]:
        """Stream from OpenAI Chat Completions API."""
        api_key = self._get_key(ModelProvider.OPENAI)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload = {
            "model": model.api_model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        self._raise_api_error("openai", model.model_id, resp.status_code, body)

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        data = json.loads(data_str)

                        # Usage chunk (final)
                        if "usage" in data and data["usage"]:
                            prompt_tokens = data["usage"].get("prompt_tokens", 0)
                            completion_tokens = data["usage"].get("completion_tokens", 0)

                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield LLMChunk(delta=content, model=model.model_id)

                    # Final chunk with token counts
                    yield LLMChunk(
                        delta="",
                        model=model.model_id,
                        is_final=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            except httpx.TimeoutException as exc:
                raise LLMError(
                    "OpenAI request timed out",
                    provider="openai",
                    model=model.model_id,
                    is_retryable=True,
                    original_error=exc,
                ) from exc

    # ── Anthropic ────────────────────────────────────────────────────────

    async def _call_anthropic(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Anthropic Messages API."""
        api_key = self._get_key(ModelProvider.ANTHROPIC)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Anthropic separates system from messages
        system_text = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                api_messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": model.api_model_name,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        data = await self._post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            payload=payload,
            provider="anthropic",
            model_id=model.model_id,
        )

        content_blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        usage = data.get("usage", {})

        return LLMResponse(
            content=text,
            model=data.get("model", model.api_model_name),
            provider=ModelProvider.ANTHROPIC,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=0.0,
            cost_usd=0.0,
            metadata={"stop_reason": data.get("stop_reason")},
        )

    async def _stream_anthropic(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[LLMChunk]:
        """Stream from Anthropic Messages API (SSE)."""
        api_key = self._get_key(ModelProvider.ANTHROPIC)
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        system_text = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_text += m.content + "\n"
            else:
                api_messages.append({"role": m.role, "content": m.content})

        payload: dict[str, Any] = {
            "model": model.api_model_name,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        self._raise_api_error("anthropic", model.model_id, resp.status_code, body)

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data = json.loads(line[6:])
                        event_type = data.get("type", "")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield LLMChunk(delta=text, model=model.model_id)

                        elif event_type == "message_delta":
                            usage_data = data.get("usage", {})
                            completion_tokens = usage_data.get("output_tokens", completion_tokens)

                        elif event_type == "message_start":
                            msg = data.get("message", {})
                            usage_data = msg.get("usage", {})
                            prompt_tokens = usage_data.get("input_tokens", 0)

                    yield LLMChunk(
                        delta="",
                        model=model.model_id,
                        is_final=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            except httpx.TimeoutException as exc:
                raise LLMError(
                    "Anthropic request timed out",
                    provider="anthropic",
                    model=model.model_id,
                    is_retryable=True,
                    original_error=exc,
                ) from exc

    # ── Google (Gemini) ──────────────────────────────────────────────────

    async def _call_google(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Google Gemini generateContent API."""
        api_key = self._get_key(ModelProvider.GOOGLE)

        # Convert messages to Gemini's contents format
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model.api_model_name}:generateContent?key={api_key}"
        )

        data = await self._post(
            url,
            headers={"Content-Type": "application/json"},
            payload=payload,
            provider="google",
            model_id=model.model_id,
        )

        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})

        return LLMResponse(
            content=text,
            model=model.api_model_name,
            provider=ModelProvider.GOOGLE,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            latency_ms=0.0,
            cost_usd=0.0,
            metadata={"finish_reason": candidates[0].get("finishReason") if candidates else None},
        )

    async def _stream_google(
        self,
        model: ModelInfo,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[LLMChunk]:
        """Stream from Google Gemini streamGenerateContent API."""
        api_key = self._get_key(ModelProvider.GOOGLE)

        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = {"parts": [{"text": m.content}]}
            else:
                role = "user" if m.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m.content}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model.api_model_name}:streamGenerateContent?key={api_key}&alt=sse"
        )

        prompt_tokens = 0
        completion_tokens = 0

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        self._raise_api_error("google", model.model_id, resp.status_code, body)

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data = json.loads(line[6:])

                        # Extract text from candidates
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    yield LLMChunk(delta=text, model=model.model_id)

                        # Token counts
                        usage = data.get("usageMetadata", {})
                        if "promptTokenCount" in usage:
                            prompt_tokens = usage["promptTokenCount"]
                        if "candidatesTokenCount" in usage:
                            completion_tokens = usage["candidatesTokenCount"]

                    yield LLMChunk(
                        delta="",
                        model=model.model_id,
                        is_final=True,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
            except httpx.TimeoutException as exc:
                raise LLMError(
                    "Google request timed out",
                    provider="google",
                    model=model.model_id,
                    is_retryable=True,
                    original_error=exc,
                ) from exc

    # ── Helpers ──────────────────────────────────────────────────────────

    def _validate_api_key(self, provider: ModelProvider) -> None:
        """Ensure the required API key is configured."""
        key = self._get_key_or_none(provider)
        if key is None:
            raise LLMError(
                f"API key not configured for {provider}. "
                f"Set the corresponding environment variable.",
                provider=provider,
            )

    def _get_key(self, provider: ModelProvider) -> str:
        """Get API key, raising LLMError if not set."""
        key = self._get_key_or_none(provider)
        if key is None:
            raise LLMError(f"Missing API key for {provider}", provider=provider)
        return key

    def _get_key_or_none(self, provider: ModelProvider) -> str | None:
        """Get API key or None if not configured."""
        if provider == ModelProvider.OPENAI and self._settings.openai_api_key:
            return self._settings.openai_api_key.get_secret_value()
        if provider == ModelProvider.ANTHROPIC and self._settings.anthropic_api_key:
            return self._settings.anthropic_api_key.get_secret_value()
        if provider == ModelProvider.GOOGLE and self._settings.google_api_key:
            return self._settings.google_api_key.get_secret_value()
        return None

    async def _post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        provider: str,
        model_id: str,
    ) -> dict[str, Any]:
        """Make a POST request and return parsed JSON, with unified error handling."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code != 200:
                self._raise_api_error(provider, model_id, resp.status_code, resp.content)

            return resp.json()

        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{provider} request timed out",
                provider=provider,
                model=model_id,
                is_retryable=True,
                original_error=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{provider} HTTP error: {exc}",
                provider=provider,
                model=model_id,
                is_retryable=True,
                original_error=exc,
            ) from exc

    @staticmethod
    def _raise_api_error(
        provider: str, model_id: str, status_code: int, body: bytes
    ) -> None:
        """Parse an API error response and raise LLMError."""
        try:
            data = json.loads(body)
            # Each provider formats errors differently
            if provider == "openai":
                msg = data.get("error", {}).get("message", str(data))
            elif provider == "anthropic":
                msg = data.get("error", {}).get("message", str(data))
            elif provider == "google":
                errors = data.get("error", {})
                msg = errors.get("message", str(data)) if isinstance(errors, dict) else str(data)
            else:
                msg = str(data)
        except (json.JSONDecodeError, AttributeError):
            msg = body.decode("utf-8", errors="replace")[:500]

        is_retryable = status_code in {429, 500, 502, 503}

        raise LLMError(
            f"{provider} API error ({status_code}): {msg}",
            provider=provider,
            model=model_id,
            status_code=status_code,
            is_retryable=is_retryable,
        )
