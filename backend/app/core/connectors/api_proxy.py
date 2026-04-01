"""OpenAI/Claude/Gemini API proxy with transparent message capture.

The API Proxy connector sits between the user's application and the
LLM provider. It forwards requests transparently while capturing
both the request and response for context building.

Flow:
    User App -> Kangaroo Shift Proxy -> OpenAI/Claude/Gemini
    User App <- Kangaroo Shift Proxy <- OpenAI/Claude/Gemini
                    |
                    v
              Capture & Process -> UCS

This is the "install and forget" ingestion path for developers.
Point your SDK at Kangaroo Shift's proxy URL instead of the
provider's URL, and context is captured automatically.

NOTE: This module handles request/response formatting and capture.
The actual HTTP forwarding is done by the API endpoint layer
(api/v1/proxy.py) which handles auth, rate limiting, and streaming.

Usage:
    from app.core.connectors.api_proxy import ProxyCapture, CapturedExchange

    capture = ProxyCapture()
    exchange = capture.capture_openai(request_body, response_body)
    ucs = exchange.generation.ucs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.core.engine.ucs_generator import GenerationResult, UCSGeneratorPipeline


@dataclass(frozen=True)
class CapturedExchange:
    """A captured API request/response pair with generated UCS.

    Attributes:
        exchange_id: Unique ID for this exchange.
        provider: LLM provider (openai, anthropic, google).
        model: Specific model used.
        request_messages: Messages sent in the request.
        response_content: Content returned in the response.
        generation: Full UCS generation result.
        captured_at: When the exchange was captured.
        request_tokens: Tokens in the request (if reported by API).
        response_tokens: Tokens in the response (if reported by API).
    """

    exchange_id: UUID
    provider: str
    model: str
    request_messages: list[dict[str, Any]]
    response_content: str
    generation: GenerationResult
    captured_at: datetime
    request_tokens: int
    response_tokens: int


class ProxyCapture:
    """Capture and process LLM API exchanges transparently.

    Detects the provider format from the request/response structure
    and runs the UCS generation pipeline on the combined messages.
    """

    def __init__(
        self,
        target_tokens: int = 4000,
        enable_spacy: bool = True,
    ) -> None:
        self._pipeline = UCSGeneratorPipeline(
            target_tokens=target_tokens,
            enable_spacy=enable_spacy,
        )

    def capture_openai(
        self,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
    ) -> CapturedExchange:
        """Capture an OpenAI Chat Completions exchange.

        Args:
            request_body: The request sent to OpenAI.
            response_body: The response from OpenAI.

        Returns:
            CapturedExchange with generated UCS.
        """
        messages = list(request_body.get("messages", []))
        model = request_body.get("model", "")

        # Extract response message
        choices = response_body.get("choices", [])
        if choices:
            resp_msg = choices[0].get("message", {})
            messages.append(resp_msg)

        # Token usage
        usage = response_body.get("usage", {})
        req_tokens = usage.get("prompt_tokens", 0)
        resp_tokens = usage.get("completion_tokens", 0)

        response_content = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            response_content = content if isinstance(content, str) else str(content)

        generation = self._pipeline.generate(messages)

        return CapturedExchange(
            exchange_id=uuid4(),
            provider="openai",
            model=model,
            request_messages=list(request_body.get("messages", [])),
            response_content=response_content,
            generation=generation,
            captured_at=datetime.now(timezone.utc),
            request_tokens=req_tokens,
            response_tokens=resp_tokens,
        )

    def capture_claude(
        self,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
    ) -> CapturedExchange:
        """Capture an Anthropic Claude Messages exchange.

        Args:
            request_body: The request sent to Claude.
            response_body: The response from Claude.

        Returns:
            CapturedExchange with generated UCS.
        """
        model = request_body.get("model", "")

        # Build combined message structure for parser
        combined: dict[str, Any] = {
            "messages": list(request_body.get("messages", [])),
        }

        # Add system prompt if present
        system = request_body.get("system", "")
        if system:
            combined["system"] = system

        # Add response as assistant message
        response_content_blocks = response_body.get("content", [])
        if response_content_blocks:
            combined["messages"].append({
                "role": "assistant",
                "content": response_content_blocks,
            })

        # Token usage
        usage = response_body.get("usage", {})
        req_tokens = usage.get("input_tokens", 0)
        resp_tokens = usage.get("output_tokens", 0)

        # Extract text from response
        response_text = ""
        for block in response_content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                response_text += block.get("text", "")

        generation = self._pipeline.generate(combined)

        return CapturedExchange(
            exchange_id=uuid4(),
            provider="anthropic",
            model=model,
            request_messages=list(request_body.get("messages", [])),
            response_content=response_text,
            generation=generation,
            captured_at=datetime.now(timezone.utc),
            request_tokens=req_tokens,
            response_tokens=resp_tokens,
        )

    def capture_gemini(
        self,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
    ) -> CapturedExchange:
        """Capture a Google Gemini generateContent exchange.

        Args:
            request_body: The request sent to Gemini.
            response_body: The response from Gemini.

        Returns:
            CapturedExchange with generated UCS.
        """
        model = request_body.get("model", "")

        # Build combined structure
        contents = list(request_body.get("contents", []))

        # Add response as model message
        candidates = response_body.get("candidates", [])
        if candidates:
            resp_content = candidates[0].get("content", {})
            if resp_content:
                contents.append(resp_content)

        combined: dict[str, Any] = {"contents": contents}

        # System instruction
        sys_instr = request_body.get("systemInstruction",
                                     request_body.get("system_instruction"))
        if sys_instr:
            combined["systemInstruction"] = sys_instr

        # Token usage
        usage_meta = response_body.get("usageMetadata", {})
        req_tokens = usage_meta.get("promptTokenCount", 0)
        resp_tokens = usage_meta.get("candidatesTokenCount", 0)

        # Extract response text
        response_text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    response_text += part["text"]

        generation = self._pipeline.generate(combined)

        return CapturedExchange(
            exchange_id=uuid4(),
            provider="google",
            model=model,
            request_messages=list(request_body.get("contents", [])),
            response_content=response_text,
            generation=generation,
            captured_at=datetime.now(timezone.utc),
            request_tokens=req_tokens,
            response_tokens=resp_tokens,
        )

    def capture_auto(
        self,
        request_body: dict[str, Any],
        response_body: dict[str, Any],
    ) -> CapturedExchange:
        """Auto-detect provider and capture the exchange.

        Detection heuristics:
            - "messages" key + "choices" in response -> OpenAI
            - "messages" key + "content" list in response -> Claude
            - "contents" key -> Gemini

        Args:
            request_body: API request body.
            response_body: API response body.

        Returns:
            CapturedExchange with generated UCS.

        Raises:
            ValueError: If provider cannot be detected.
        """
        # OpenAI: has "messages" + response has "choices"
        if "messages" in request_body and "choices" in response_body:
            return self.capture_openai(request_body, response_body)

        # Claude: has "messages" + response has "content" list + "role"
        if "messages" in request_body and isinstance(
            response_body.get("content"), list
        ):
            return self.capture_claude(request_body, response_body)

        # Gemini: has "contents"
        if "contents" in request_body:
            return self.capture_gemini(request_body, response_body)

        raise ValueError(
            "Cannot auto-detect API provider. "
            "Use capture_openai(), capture_claude(), or capture_gemini() directly."
        )
