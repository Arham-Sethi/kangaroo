"""Tests for ingestion connectors (manual import + API proxy capture).

Tests cover:
    1. Manual import (text, JSON, file)
    2. API proxy capture (OpenAI, Claude, Gemini, auto-detect)
    3. Error handling
    4. Format auto-detection

Total: 35+ tests
"""

from __future__ import annotations

import json

import pytest

from app.core.connectors.manual_import import ImportResult, ManualImportConnector
from app.core.connectors.api_proxy import CapturedExchange, ProxyCapture


# -- Shared fixtures ---------------------------------------------------------

_OPENAI_MESSAGES = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
]

_CLAUDE_REQUEST = {
    "model": "claude-sonnet-4-20250514",
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "What is Python?"}]},
    ],
}

_CLAUDE_RESPONSE = {
    "content": [{"type": "text", "text": "Python is a programming language."}],
    "model": "claude-sonnet-4-20250514",
    "usage": {"input_tokens": 10, "output_tokens": 20},
}

_GEMINI_REQUEST = {
    "contents": [
        {"role": "user", "parts": [{"text": "What is Python?"}]},
    ],
}

_GEMINI_RESPONSE = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "Python is a language."}]}},
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15},
}


# == Manual Import Tests =====================================================


class TestManualImportConnector:
    def setup_method(self) -> None:
        self.connector = ManualImportConnector(enable_spacy=False)

    # -- Text import --

    def test_import_text_simple(self) -> None:
        result = self.connector.import_text("User: Hello\nAssistant: Hi there!")
        assert isinstance(result, ImportResult)
        assert result.source_type == "text_plain"
        assert result.original_size_bytes > 0

    def test_import_text_json_string(self) -> None:
        json_str = json.dumps(_OPENAI_MESSAGES)
        result = self.connector.import_text(json_str)
        assert result.source_type == "text_json"
        assert result.detected_format in ("openai", "generic")

    def test_import_text_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.connector.import_text("")

    def test_import_text_whitespace_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.connector.import_text("   ")

    def test_import_text_plain(self) -> None:
        result = self.connector.import_text("Just a plain message")
        assert result.source_type == "text_plain"
        assert result.detected_format == "generic"

    # -- JSON import --

    def test_import_json_openai(self) -> None:
        result = self.connector.import_json(_OPENAI_MESSAGES)
        assert isinstance(result, ImportResult)
        assert result.source_type == "json"

    def test_import_json_dict(self) -> None:
        # OpenAI API response format (dict with "choices")
        data = {
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "model": "gpt-4o",
        }
        result = self.connector.import_json(data)
        assert result.source_type == "json"

    def test_import_json_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.connector.import_json([])

    def test_import_json_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.connector.import_json({})

    # -- File import --

    def test_import_file_json(self) -> None:
        content = json.dumps(_OPENAI_MESSAGES).encode("utf-8")
        result = self.connector.import_file(content, filename="chat.json")
        assert result.source_type == "file_json"

    def test_import_file_txt(self) -> None:
        content = b"User: Hello\nAssistant: Hi!"
        result = self.connector.import_file(content, filename="chat.txt")
        assert result.source_type == "file_text"

    def test_import_file_md(self) -> None:
        content = b"User: Hello\nAssistant: Hi!"
        result = self.connector.import_file(content, filename="conversation.md")
        assert result.source_type == "file_text"

    def test_import_file_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            self.connector.import_file(b"", filename="empty.json")

    def test_import_file_no_extension(self) -> None:
        content = b"User: Hello\nAssistant: Hi!"
        result = self.connector.import_file(content, filename="noext")
        assert result.source_type == "file_text"

    def test_import_file_unsupported_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            self.connector.import_file(b"data", filename="file.xyz")

    def test_import_file_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            self.connector.import_file(b"not json {{{", filename="bad.json")

    # -- UCS quality --

    def test_imported_ucs_has_entities(self) -> None:
        text = "We use Python with FastAPI and PostgreSQL for the backend"
        result = self.connector.import_text(text)
        gen = result.generation
        assert len(gen.ucs.entities) >= 1

    def test_imported_ucs_has_summaries(self) -> None:
        text = "User: Build me a REST API\nAssistant: Sure, let's use FastAPI with PostgreSQL"
        result = self.connector.import_text(text)
        gen = result.generation
        assert len(gen.ucs.summaries) >= 1


# == API Proxy Capture Tests =================================================


class TestProxyCapture:
    def setup_method(self) -> None:
        self.capture = ProxyCapture(enable_spacy=False)

    # -- OpenAI capture --

    def test_capture_openai(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Reply"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = self.capture.capture_openai(request, response)
        assert isinstance(result, CapturedExchange)
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.request_tokens == 10
        assert result.response_tokens == 5

    def test_capture_openai_ucs_generated(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Python is great"}}],
            "usage": {},
        }
        result = self.capture.capture_openai(request, response)
        assert result.generation.ucs is not None

    # -- Claude capture --

    def test_capture_claude(self) -> None:
        result = self.capture.capture_claude(_CLAUDE_REQUEST, _CLAUDE_RESPONSE)
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.request_tokens == 10
        assert result.response_tokens == 20

    def test_capture_claude_with_system(self) -> None:
        request = {**_CLAUDE_REQUEST, "system": "You are a helpful assistant."}
        result = self.capture.capture_claude(request, _CLAUDE_RESPONSE)
        assert result.generation.ucs is not None

    # -- Gemini capture --

    def test_capture_gemini(self) -> None:
        result = self.capture.capture_gemini(_GEMINI_REQUEST, _GEMINI_RESPONSE)
        assert result.provider == "google"
        assert result.request_tokens == 10
        assert result.response_tokens == 15

    # -- Auto-detect --

    def test_auto_detect_openai(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {
            "choices": [{"message": {"role": "assistant", "content": "Reply"}}],
            "usage": {},
        }
        result = self.capture.capture_auto(request, response)
        assert result.provider == "openai"

    def test_auto_detect_claude(self) -> None:
        result = self.capture.capture_auto(_CLAUDE_REQUEST, _CLAUDE_RESPONSE)
        assert result.provider == "anthropic"

    def test_auto_detect_gemini(self) -> None:
        result = self.capture.capture_auto(_GEMINI_REQUEST, _GEMINI_RESPONSE)
        assert result.provider == "google"

    def test_auto_detect_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            self.capture.capture_auto({"unknown": True}, {"also_unknown": True})

    # -- Exchange metadata --

    def test_exchange_has_id(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {"choices": [{"message": {"content": "Hi"}}], "usage": {}}
        result = self.capture.capture_openai(request, response)
        assert result.exchange_id is not None

    def test_exchange_has_timestamp(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {"choices": [{"message": {"content": "Hi"}}], "usage": {}}
        result = self.capture.capture_openai(request, response)
        assert result.captured_at is not None

    def test_exchange_response_content(self) -> None:
        request = {"model": "gpt-4o", "messages": _OPENAI_MESSAGES}
        response = {
            "choices": [{"message": {"role": "assistant", "content": "The answer is 42"}}],
            "usage": {},
        }
        result = self.capture.capture_openai(request, response)
        assert result.response_content == "The answer is 42"
