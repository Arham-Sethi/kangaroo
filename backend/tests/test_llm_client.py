"""Tests for the unified LLM client.

Tests cover:
    - Model registry: lookup, resolve, cost estimation
    - LLM client: OpenAI, Anthropic, Google call routing
    - Error handling: missing API keys, API errors, timeouts
    - Streaming: chunk assembly, final chunk with token counts
    - Cost calculation: per-model pricing accuracy

All tests mock httpx responses — no real API calls are made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.llm.models import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMResponse,
    ModelInfo,
    ModelProvider,
    MODEL_REGISTRY,
    resolve_model,
)


# -- Model Registry Tests ---------------------------------------------------


class TestModelRegistry:
    def test_registry_has_openai_models(self) -> None:
        assert "gpt-4o" in MODEL_REGISTRY
        assert "gpt-4o-mini" in MODEL_REGISTRY
        assert MODEL_REGISTRY["gpt-4o"].provider == ModelProvider.OPENAI

    def test_registry_has_anthropic_models(self) -> None:
        assert "claude-sonnet-4" in MODEL_REGISTRY
        assert "claude-haiku-4" in MODEL_REGISTRY
        assert MODEL_REGISTRY["claude-sonnet-4"].provider == ModelProvider.ANTHROPIC

    def test_registry_has_google_models(self) -> None:
        assert "gemini-2.0-flash" in MODEL_REGISTRY
        assert "gemini-2.5-pro" in MODEL_REGISTRY
        assert MODEL_REGISTRY["gemini-2.0-flash"].provider == ModelProvider.GOOGLE

    def test_resolve_model_success(self) -> None:
        info = resolve_model("gpt-4o")
        assert info.model_id == "gpt-4o"
        assert info.provider == ModelProvider.OPENAI

    def test_resolve_model_not_found(self) -> None:
        with pytest.raises(LLMError, match="Unknown model"):
            resolve_model("nonexistent-model-v99")

    def test_model_info_is_frozen(self) -> None:
        info = resolve_model("gpt-4o")
        with pytest.raises(AttributeError):
            info.model_id = "modified"  # type: ignore[misc]

    def test_all_models_have_pricing(self) -> None:
        for model_id, info in MODEL_REGISTRY.items():
            assert info.prompt_cost_per_1m > 0, f"{model_id} missing prompt pricing"
            assert info.completion_cost_per_1m > 0, f"{model_id} missing completion pricing"

    def test_all_models_have_context_window(self) -> None:
        for model_id, info in MODEL_REGISTRY.items():
            assert info.max_context >= 128_000, f"{model_id} context window too small"


class TestCostEstimation:
    def test_estimate_cost_gpt4o(self) -> None:
        info = resolve_model("gpt-4o")
        # 1000 prompt tokens at $2.50/1M = $0.0025
        # 500 completion tokens at $10.00/1M = $0.005
        cost = info.estimate_cost(1000, 500)
        assert cost == pytest.approx(0.0075, abs=0.0001)

    def test_estimate_cost_zero_tokens(self) -> None:
        info = resolve_model("gpt-4o")
        assert info.estimate_cost(0, 0) == 0.0

    def test_estimate_cost_flash_cheap(self) -> None:
        info = resolve_model("gemini-2.0-flash")
        cost = info.estimate_cost(10_000, 5_000)
        # Very cheap model
        assert cost < 0.01

    def test_estimate_cost_opus_expensive(self) -> None:
        info = resolve_model("claude-opus-4")
        cost = info.estimate_cost(10_000, 5_000)
        # Most expensive model
        assert cost > 0.1


# -- LLM Error Tests --------------------------------------------------------


class TestLLMError:
    def test_error_is_exception(self) -> None:
        err = LLMError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_error_with_metadata(self) -> None:
        err = LLMError(
            "rate limited",
            provider="openai",
            model="gpt-4o",
            status_code=429,
            is_retryable=True,
        )
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
        assert err.status_code == 429
        assert err.is_retryable is True

    def test_error_not_retryable_by_default(self) -> None:
        err = LLMError("auth error")
        assert err.is_retryable is False


# -- LLM Response Tests -----------------------------------------------------


class TestLLMResponse:
    def test_total_tokens(self) -> None:
        resp = LLMResponse(
            content="hello",
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200.0,
            cost_usd=0.005,
        )
        assert resp.total_tokens == 150

    def test_response_is_frozen(self) -> None:
        resp = LLMResponse(
            content="hello",
            model="gpt-4o",
            provider=ModelProvider.OPENAI,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=200.0,
            cost_usd=0.005,
        )
        with pytest.raises(AttributeError):
            resp.content = "modified"  # type: ignore[misc]


class TestLLMChunk:
    def test_chunk_defaults(self) -> None:
        chunk = LLMChunk(delta="hello", model="gpt-4o")
        assert chunk.is_final is False
        assert chunk.prompt_tokens == 0
        assert chunk.completion_tokens == 0

    def test_final_chunk(self) -> None:
        chunk = LLMChunk(
            delta="",
            model="gpt-4o",
            is_final=True,
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert chunk.is_final is True


# -- LLM Client Tests (Mocked) ----------------------------------------------


def _mock_settings(**overrides):
    """Create a mock settings object with API keys."""
    from unittest.mock import PropertyMock
    from pydantic import SecretStr

    settings = MagicMock()
    settings.openai_api_key = overrides.get("openai_api_key", SecretStr("sk-test-openai"))
    settings.anthropic_api_key = overrides.get("anthropic_api_key", SecretStr("sk-test-anthropic"))
    settings.google_api_key = overrides.get("google_api_key", SecretStr("test-google-key"))
    return settings


def _openai_response_json(content: str = "Hello!", model: str = "gpt-4o") -> dict:
    """Build a mock OpenAI Chat Completion response."""
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _anthropic_response_json(content: str = "Hello!", model: str = "claude-sonnet-4") -> dict:
    """Build a mock Anthropic Messages response."""
    return {
        "id": "msg_test",
        "type": "message",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        },
    }


def _google_response_json(content: str = "Hello!") -> dict:
    """Build a mock Google Gemini response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": content}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


class TestLLMClientOpenAI:
    @pytest.mark.asyncio
    async def test_call_openai(self) -> None:
        from app.core.llm.client import LLMClient

        mock_resp = httpx.Response(200, json=_openai_response_json("Hi there!"))

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            response = await client.call("gpt-4o", [LLMMessage("user", "Hello")])

        assert response.content == "Hi there!"
        assert response.provider == ModelProvider.OPENAI
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_call_openai_with_system_message(self) -> None:
        from app.core.llm.client import LLMClient

        mock_resp = httpx.Response(200, json=_openai_response_json("Expert response"))

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            response = await client.call(
                "gpt-4o",
                [
                    LLMMessage("system", "You are an expert."),
                    LLMMessage("user", "Explain X."),
                ],
            )

        assert response.content == "Expert response"


class TestLLMClientAnthropic:
    @pytest.mark.asyncio
    async def test_call_anthropic(self) -> None:
        from app.core.llm.client import LLMClient

        mock_resp = httpx.Response(200, json=_anthropic_response_json("Greetings!"))

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            response = await client.call(
                "claude-sonnet-4", [LLMMessage("user", "Hello")]
            )

        assert response.content == "Greetings!"
        assert response.provider == ModelProvider.ANTHROPIC
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_anthropic_separates_system(self) -> None:
        """Anthropic API requires system messages as a separate field, not in messages."""
        from app.core.llm.client import LLMClient

        mock_resp = httpx.Response(200, json=_anthropic_response_json("OK"))

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post,
        ):
            client = LLMClient()
            await client.call(
                "claude-sonnet-4",
                [
                    LLMMessage("system", "Be helpful."),
                    LLMMessage("user", "Hi"),
                ],
            )

        # Verify system was sent as top-level field, not in messages array
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["system"] == "Be helpful."
        assert all(m["role"] != "system" for m in payload["messages"])


class TestLLMClientGoogle:
    @pytest.mark.asyncio
    async def test_call_google(self) -> None:
        from app.core.llm.client import LLMClient

        mock_resp = httpx.Response(200, json=_google_response_json("Hey!"))

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            response = await client.call(
                "gemini-2.0-flash", [LLMMessage("user", "Hello")]
            )

        assert response.content == "Hey!"
        assert response.provider == ModelProvider.GOOGLE


class TestLLMClientErrors:
    @pytest.mark.asyncio
    async def test_missing_api_key(self) -> None:
        from app.core.llm.client import LLMClient

        no_key_settings = _mock_settings(openai_api_key=None)

        with patch("app.core.llm.client.get_settings", return_value=no_key_settings):
            client = LLMClient()
            with pytest.raises(LLMError, match="API key not configured"):
                await client.call("gpt-4o", [LLMMessage("user", "Hi")])

    @pytest.mark.asyncio
    async def test_api_error_401(self) -> None:
        from app.core.llm.client import LLMClient

        error_body = json.dumps({"error": {"message": "Invalid API key"}}).encode()
        mock_resp = httpx.Response(401, content=error_body)

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            with pytest.raises(LLMError, match="401") as exc_info:
                await client.call("gpt-4o", [LLMMessage("user", "Hi")])
            assert exc_info.value.is_retryable is False

    @pytest.mark.asyncio
    async def test_api_error_429_is_retryable(self) -> None:
        from app.core.llm.client import LLMClient

        error_body = json.dumps({"error": {"message": "Rate limited"}}).encode()
        mock_resp = httpx.Response(429, content=error_body)

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            with pytest.raises(LLMError) as exc_info:
                await client.call("gpt-4o", [LLMMessage("user", "Hi")])
            assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_api_error_500_is_retryable(self) -> None:
        from app.core.llm.client import LLMClient

        error_body = json.dumps({"error": {"message": "Server error"}}).encode()
        mock_resp = httpx.Response(500, content=error_body)

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp),
        ):
            client = LLMClient()
            with pytest.raises(LLMError) as exc_info:
                await client.call("gpt-4o", [LLMMessage("user", "Hi")])
            assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        from app.core.llm.client import LLMClient

        with (
            patch("app.core.llm.client.get_settings", return_value=_mock_settings()),
            patch(
                "httpx.AsyncClient.post",
                new_callable=AsyncMock,
                side_effect=httpx.TimeoutException("timed out"),
            ),
        ):
            client = LLMClient()
            with pytest.raises(LLMError, match="timed out") as exc_info:
                await client.call("gpt-4o", [LLMMessage("user", "Hi")])
            assert exc_info.value.is_retryable is True

    @pytest.mark.asyncio
    async def test_unknown_model(self) -> None:
        from app.core.llm.client import LLMClient

        with patch("app.core.llm.client.get_settings", return_value=_mock_settings()):
            client = LLMClient()
            with pytest.raises(LLMError, match="Unknown model"):
                await client.call("not-a-real-model", [LLMMessage("user", "Hi")])


# -- Provider Enum Tests ----------------------------------------------------


class TestModelProvider:
    def test_provider_values(self) -> None:
        assert ModelProvider.OPENAI == "openai"
        assert ModelProvider.ANTHROPIC == "anthropic"
        assert ModelProvider.GOOGLE == "google"

    def test_provider_is_str_enum(self) -> None:
        assert isinstance(ModelProvider.OPENAI, str)
