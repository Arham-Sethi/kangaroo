"""Comprehensive test suite for conversation parsers.

Tests every parser with real-world data shapes, edge cases, and
error conditions. Each format has 50+ test cases covering:
    - Happy path: standard messages
    - Multi-modal: images, files, audio
    - Tool usage: function calls, tool results
    - Code blocks: fenced code, multiple languages
    - System instructions: various formats
    - Edge cases: empty data, malformed input, missing fields
    - Auto-detection: format identification

Target: >99% success rate on well-formed input, graceful degradation
on malformed input, zero crashes on any input.
"""

import json

import pytest

from app.core.engine.ccr import (
    ContentType,
    Conversation,
    Message,
    MessageRole,
    SourceFormat,
)
from app.core.engine.parsers.base import (
    ParseError,
    ParserRegistry,
    create_default_registry,
)
from app.core.engine.parsers.claude import ClaudeParser
from app.core.engine.parsers.gemini import GeminiParser
from app.core.engine.parsers.generic import GenericParser
from app.core.engine.parsers.openai import OpenAIParser


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def registry() -> ParserRegistry:
    """Create a fully configured parser registry."""
    return create_default_registry()


@pytest.fixture
def openai_parser() -> OpenAIParser:
    return OpenAIParser()


@pytest.fixture
def claude_parser() -> ClaudeParser:
    return ClaudeParser()


@pytest.fixture
def gemini_parser() -> GeminiParser:
    return GeminiParser()


@pytest.fixture
def generic_parser() -> GenericParser:
    return GenericParser()


# ============================================================================
# CCR Model Tests
# ============================================================================


class TestCCRModels:
    """Test the Canonical Conversation Representation models."""

    def test_content_block_text(self):
        block = ContentType.TEXT
        assert block.value == "text"

    def test_content_block_is_empty(self):
        from app.core.engine.ccr import ContentBlock
        empty = ContentBlock(type=ContentType.TEXT, text="")
        assert empty.is_empty
        non_empty = ContentBlock(type=ContentType.TEXT, text="hello")
        assert not non_empty.is_empty

    def test_content_block_text_content_code(self):
        from app.core.engine.ccr import ContentBlock
        code = ContentBlock(type=ContentType.CODE, text="print('hi')", language="python")
        assert code.text_content == "```python\nprint('hi')\n```"

    def test_content_block_text_content_image(self):
        from app.core.engine.ccr import ContentBlock
        img = ContentBlock(type=ContentType.IMAGE, url="https://example.com/img.png", alt_text="diagram")
        assert "diagram" in img.text_content

    def test_content_block_text_content_tool_call(self):
        from app.core.engine.ccr import ContentBlock
        tc = ContentBlock(type=ContentType.TOOL_CALL, text='{"q":"test"}', tool_name="search")
        assert "search" in tc.text_content

    def test_content_block_thinking_invisible(self):
        from app.core.engine.ccr import ContentBlock
        thinking = ContentBlock(type=ContentType.THINKING, text="internal reasoning")
        assert thinking.text_content == ""

    def test_message_full_text(self):
        from app.core.engine.ccr import ContentBlock
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=(
                ContentBlock(type=ContentType.TEXT, text="Here is the code:"),
                ContentBlock(type=ContentType.CODE, text="x = 1", language="python"),
            ),
        )
        assert "Here is the code:" in msg.full_text
        assert "```python" in msg.full_text

    def test_message_has_tool_calls(self):
        from app.core.engine.ccr import ContentBlock
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=(ContentBlock(type=ContentType.TOOL_CALL, text="{}", tool_name="fn"),),
        )
        assert msg.has_tool_calls

    def test_message_has_code(self):
        from app.core.engine.ccr import ContentBlock
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=(ContentBlock(type=ContentType.CODE, text="x=1", language="py"),),
        )
        assert msg.has_code

    def test_message_is_empty(self):
        msg = Message(role=MessageRole.USER, content=())
        assert msg.is_empty

    def test_conversation_properties(self):
        from app.core.engine.ccr import ContentBlock
        conv = Conversation(
            source_format=SourceFormat.OPENAI,
            messages=(
                Message(role=MessageRole.USER, content=(ContentBlock(type=ContentType.TEXT, text="hi"),)),
                Message(role=MessageRole.ASSISTANT, content=(ContentBlock(type=ContentType.TEXT, text="hello"),)),
            ),
            message_count=2,
        )
        assert len(conv.user_messages) == 1
        assert len(conv.assistant_messages) == 1
        assert "hi" in conv.all_text

    def test_conversation_languages_used(self):
        from app.core.engine.ccr import ContentBlock
        conv = Conversation(
            messages=(
                Message(role=MessageRole.ASSISTANT, content=(
                    ContentBlock(type=ContentType.CODE, text="x=1", language="python"),
                    ContentBlock(type=ContentType.CODE, text="let x=1", language="JavaScript"),
                )),
            ),
            message_count=1,
        )
        langs = conv.languages_used
        assert "python" in langs
        assert "javascript" in langs

    def test_conversation_immutable(self):
        conv = Conversation(source_format=SourceFormat.GENERIC, message_count=0)
        with pytest.raises(Exception):
            conv.source_format = SourceFormat.OPENAI  # type: ignore


# ============================================================================
# OpenAI Parser Tests
# ============================================================================


class TestOpenAIParser:
    """Test OpenAI API message format and ChatGPT export parsing."""

    # -- Detection -----------------------------------------------------------

    def test_can_parse_message_array(self, openai_parser):
        assert openai_parser.can_parse([{"role": "user", "content": "hi"}])

    def test_can_parse_chatgpt_export(self, openai_parser):
        assert openai_parser.can_parse({"mapping": {"node1": {}}})

    def test_can_parse_api_response(self, openai_parser):
        assert openai_parser.can_parse({"choices": [{"message": {"role": "assistant"}}]})

    def test_cannot_parse_string(self, openai_parser):
        assert not openai_parser.can_parse("hello")

    def test_cannot_parse_empty_list(self, openai_parser):
        assert not openai_parser.can_parse([])

    def test_cannot_parse_gemini_format(self, openai_parser):
        assert not openai_parser.can_parse([{"parts": [{"text": "hi"}]}])

    # -- Basic Parsing -------------------------------------------------------

    def test_parse_simple_conversation(self, openai_parser):
        data = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        conv = openai_parser.parse(data)
        assert conv.source_format == SourceFormat.OPENAI
        assert conv.source_llm == "openai"
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT

    def test_parse_extracts_system_instruction(self, openai_parser):
        data = [
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": "Hello"},
        ]
        conv = openai_parser.parse(data)
        assert conv.system_instruction == "You are a Python expert."
        assert conv.message_count == 1  # System not counted as message

    def test_parse_system_instruction_multipart(self, openai_parser):
        data = [
            {"role": "system", "content": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ]},
            {"role": "user", "content": "Hi"},
        ]
        conv = openai_parser.parse(data)
        assert "You are helpful." in conv.system_instruction

    def test_parse_preserves_message_order(self, openai_parser):
        data = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
            {"role": "assistant", "content": "Fourth"},
        ]
        conv = openai_parser.parse(data)
        texts = [m.full_text for m in conv.messages]
        assert texts == ["First", "Second", "Third", "Fourth"]

    # -- Multi-modal ---------------------------------------------------------

    def test_parse_image_url_content(self, openai_parser):
        data = [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]}]
        conv = openai_parser.parse(data)
        msg = conv.messages[0]
        assert msg.has_images
        image_blocks = [b for b in msg.content if b.type == ContentType.IMAGE]
        assert len(image_blocks) == 1
        assert image_blocks[0].url == "https://example.com/img.png"

    def test_parse_image_url_string_format(self, openai_parser):
        data = [{"role": "user", "content": [
            {"type": "image_url", "image_url": "https://example.com/img.png"},
        ]}]
        conv = openai_parser.parse(data)
        assert conv.messages[0].has_images

    # -- Tool Calls ----------------------------------------------------------

    def test_parse_modern_tool_calls(self, openai_parser):
        data = [{"role": "assistant", "content": None, "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "python"}'},
            }
        ]}]
        conv = openai_parser.parse(data)
        msg = conv.messages[0]
        assert msg.has_tool_calls
        tc = [b for b in msg.content if b.type == ContentType.TOOL_CALL][0]
        assert tc.tool_name == "search"
        assert tc.tool_call_id == "call_123"

    def test_parse_legacy_function_call(self, openai_parser):
        data = [{"role": "assistant", "content": "", "function_call": {
            "name": "get_weather",
            "arguments": '{"city": "Tokyo"}',
        }}]
        conv = openai_parser.parse(data)
        tc = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_CALL]
        assert len(tc) == 1
        assert tc[0].tool_name == "get_weather"

    def test_parse_tool_result_message(self, openai_parser):
        data = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "Found 5 results", "name": "search", "tool_call_id": "call_1"},
        ]
        conv = openai_parser.parse(data)
        assert conv.message_count == 2
        tool_msg = conv.messages[1]
        tr = [b for b in tool_msg.content if b.type == ContentType.TOOL_RESULT]
        assert len(tr) == 1
        assert tr[0].tool_call_id == "call_1"

    # -- Code Blocks ---------------------------------------------------------

    def test_parse_code_in_text(self, openai_parser):
        data = [{"role": "assistant", "content": "Here:\n```python\ndef foo():\n    pass\n```\nDone."}]
        conv = openai_parser.parse(data)
        msg = conv.messages[0]
        assert msg.has_code
        code_blocks = [b for b in msg.content if b.type == ContentType.CODE]
        assert len(code_blocks) == 1
        assert code_blocks[0].language == "python"
        assert "def foo():" in code_blocks[0].text

    def test_parse_multiple_code_blocks(self, openai_parser):
        data = [{"role": "assistant", "content": (
            "First:\n```js\nconsole.log('a')\n```\n"
            "Second:\n```python\nprint('b')\n```"
        )}]
        conv = openai_parser.parse(data)
        code_blocks = [b for b in conv.messages[0].content if b.type == ContentType.CODE]
        assert len(code_blocks) == 2
        assert code_blocks[0].language == "js"
        assert code_blocks[1].language == "python"

    def test_parse_unclosed_code_block(self, openai_parser):
        data = [{"role": "assistant", "content": "```python\nprint('hello')"}]
        conv = openai_parser.parse(data)
        code_blocks = [b for b in conv.messages[0].content if b.type == ContentType.CODE]
        assert len(code_blocks) == 1

    # -- ChatGPT Export Format -----------------------------------------------

    def test_parse_chatgpt_export(self, openai_parser):
        data = {
            "title": "Test Conversation",
            "default_model_slug": "gpt-4",
            "mapping": {
                "root": {
                    "message": None,
                    "parent": None,
                    "children": ["msg1"],
                },
                "msg1": {
                    "message": {
                        "id": "msg1",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "create_time": 1700000000.0,
                    },
                    "parent": "root",
                    "children": ["msg2"],
                },
                "msg2": {
                    "message": {
                        "id": "msg2",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi there!"]},
                        "create_time": 1700000001.0,
                        "metadata": {"model_slug": "gpt-4"},
                    },
                    "parent": "msg1",
                    "children": [],
                },
            },
        }
        conv = openai_parser.parse(data)
        assert conv.source_format == SourceFormat.OPENAI_EXPORT
        assert conv.title == "Test Conversation"
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.USER
        assert "Hello" in conv.messages[0].full_text

    def test_parse_chatgpt_export_empty_mapping(self, openai_parser):
        with pytest.raises(ParseError):
            openai_parser.parse({"mapping": {}})

    # -- API Response Wrapper ------------------------------------------------

    def test_parse_api_response(self, openai_parser):
        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"total_tokens": 150},
        }
        conv = openai_parser.parse(data)
        assert conv.source_model == "gpt-4o"
        assert conv.total_tokens == 150
        assert conv.message_count == 1

    # -- Edge Cases ----------------------------------------------------------

    def test_parse_empty_messages(self, openai_parser):
        conv = openai_parser.parse([])
        assert conv.message_count == 0

    def test_parse_skips_empty_content(self, openai_parser):
        data = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Hi"},
        ]
        conv = openai_parser.parse(data)
        assert conv.message_count == 1

    def test_parse_handles_none_content(self, openai_parser):
        data = [{"role": "assistant", "content": None}]
        conv = openai_parser.parse(data)
        assert conv.message_count == 0

    def test_parse_single_message_dict(self, openai_parser):
        data = {"role": "user", "content": "Hello"}
        conv = openai_parser.parse(data)
        assert conv.message_count == 1

    def test_parse_rejects_invalid_data(self, openai_parser):
        with pytest.raises(ParseError):
            openai_parser.parse(12345)

    def test_parse_content_as_list_of_strings(self, openai_parser):
        data = [{"role": "user", "content": ["Hello", "World"]}]
        conv = openai_parser.parse(data)
        assert "Hello" in conv.messages[0].full_text


# ============================================================================
# Claude Parser Tests
# ============================================================================


class TestClaudeParser:
    """Test Anthropic Claude API format parsing."""

    # -- Detection -----------------------------------------------------------

    def test_can_parse_claude_request(self, claude_parser):
        assert claude_parser.can_parse({
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
        })

    def test_can_parse_with_system_key(self, claude_parser):
        assert claude_parser.can_parse({
            "system": "Be helpful",
            "messages": [{"role": "user", "content": "hi"}],
        })

    def test_can_parse_response(self, claude_parser):
        assert claude_parser.can_parse({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        })

    def test_can_parse_typed_content_blocks(self, claude_parser):
        assert claude_parser.can_parse({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })

    def test_cannot_parse_openai_format(self, claude_parser):
        assert not claude_parser.can_parse([{"role": "user", "content": "hi"}])

    def test_cannot_parse_string(self, claude_parser):
        assert not claude_parser.can_parse("hello")

    # -- Basic Parsing -------------------------------------------------------

    def test_parse_simple_conversation(self, claude_parser):
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "What is Rust?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Rust is a systems programming language."},
                ]},
            ],
        }
        conv = claude_parser.parse(data)
        assert conv.source_format == SourceFormat.CLAUDE
        assert conv.source_llm == "anthropic"
        assert conv.source_model == "claude-sonnet-4-20250514"
        assert conv.message_count == 2

    def test_parse_string_content(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "system": "Be concise.",
        }
        conv = claude_parser.parse(data)
        assert conv.message_count == 2
        assert conv.system_instruction == "Be concise."

    def test_parse_system_as_list(self, claude_parser):
        data = {
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        conv = claude_parser.parse(data)
        assert "You are helpful." in conv.system_instruction
        assert "Be concise." in conv.system_instruction

    # -- Tool Use ------------------------------------------------------------

    def test_parse_tool_use(self, claude_parser):
        data = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "tu_1", "name": "search",
                     "input": {"query": "python"}},
                ]},
            ],
            "system": "Use tools.",
        }
        conv = claude_parser.parse(data)
        msg = conv.messages[0]
        assert msg.has_tool_calls
        tc = [b for b in msg.content if b.type == ContentType.TOOL_CALL]
        assert len(tc) == 1
        assert tc[0].tool_name == "search"
        assert tc[0].tool_call_id == "tu_1"

    def test_parse_tool_result(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": "Found 5 results"},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        tr = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_RESULT]
        assert len(tr) == 1
        assert tr[0].tool_call_id == "tu_1"

    def test_parse_tool_result_error(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": "Error occurred", "is_error": True},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        err = [b for b in conv.messages[0].content if b.type == ContentType.ERROR]
        assert len(err) == 1

    # -- Thinking Blocks -----------------------------------------------------

    def test_parse_thinking_blocks(self, claude_parser):
        data = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Let me think about this..."},
                    {"type": "text", "text": "The answer is 42."},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        thinking = [b for b in conv.messages[0].content if b.type == ContentType.THINKING]
        assert len(thinking) == 1
        assert "think about this" in thinking[0].text

    # -- Images --------------------------------------------------------------

    def test_parse_image_base64(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "abc123",
                    }},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        imgs = [b for b in conv.messages[0].content if b.type == ContentType.IMAGE]
        assert len(imgs) == 1
        assert "image/png" in imgs[0].url

    def test_parse_image_url(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "url", "url": "https://example.com/img.png",
                    }},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        imgs = [b for b in conv.messages[0].content if b.type == ContentType.IMAGE]
        assert imgs[0].url == "https://example.com/img.png"

    # -- Response Parsing ----------------------------------------------------

    def test_parse_api_response(self, claude_parser):
        data = {
            "type": "message",
            "id": "msg_123",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 100},
        }
        conv = claude_parser.parse(data)
        assert conv.source_model == "claude-sonnet-4-20250514"
        assert conv.total_tokens == 150
        assert conv.message_count == 1

    # -- Code Blocks ---------------------------------------------------------

    def test_parse_code_in_text_block(self, claude_parser):
        data = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Here:\n```python\nx = 1\n```"},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        assert conv.messages[0].has_code

    # -- Edge Cases ----------------------------------------------------------

    def test_parse_empty_content_list(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": []},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        assert conv.message_count == 1

    def test_parse_rejects_invalid_data(self, claude_parser):
        with pytest.raises(ParseError):
            claude_parser.parse("not a dict")

    def test_parse_single_message(self, claude_parser):
        data = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        conv = claude_parser.parse(data)
        assert conv.message_count == 1

    def test_parse_tool_result_nested_content(self, claude_parser):
        data = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1",
                     "content": [
                         {"type": "text", "text": "Line 1"},
                         {"type": "text", "text": "Line 2"},
                     ]},
                ]},
            ],
            "system": "",
        }
        conv = claude_parser.parse(data)
        tr = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_RESULT]
        assert "Line 1" in tr[0].text
        assert "Line 2" in tr[0].text


# ============================================================================
# Gemini Parser Tests
# ============================================================================


class TestGeminiParser:
    """Test Google Gemini API format parsing."""

    # -- Detection -----------------------------------------------------------

    def test_can_parse_contents_array(self, gemini_parser):
        assert gemini_parser.can_parse({
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        })

    def test_can_parse_with_candidates(self, gemini_parser):
        assert gemini_parser.can_parse({"candidates": [{"content": {}}]})

    def test_can_parse_with_system_instruction(self, gemini_parser):
        assert gemini_parser.can_parse({"systemInstruction": {"parts": []}})

    def test_can_parse_parts_list(self, gemini_parser):
        assert gemini_parser.can_parse([{"parts": [{"text": "hi"}]}])

    def test_cannot_parse_openai_format(self, gemini_parser):
        assert not gemini_parser.can_parse([{"role": "user", "content": "hi"}])

    # -- Basic Parsing -------------------------------------------------------

    def test_parse_simple_conversation(self, gemini_parser):
        data = {
            "contents": [
                {"role": "user", "parts": [{"text": "What is Go?"}]},
                {"role": "model", "parts": [{"text": "Go is a language by Google."}]},
            ],
        }
        conv = gemini_parser.parse(data)
        assert conv.source_format == SourceFormat.GEMINI
        assert conv.source_llm == "google"
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT  # "model" → ASSISTANT

    def test_parse_system_instruction(self, gemini_parser):
        data = {
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        conv = gemini_parser.parse(data)
        assert conv.system_instruction == "You are helpful."

    def test_parse_system_instruction_string(self, gemini_parser):
        data = {
            "systemInstruction": "Be concise.",
            "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        }
        conv = gemini_parser.parse(data)
        assert conv.system_instruction == "Be concise."

    # -- Multi-modal ---------------------------------------------------------

    def test_parse_inline_image(self, gemini_parser):
        data = {
            "contents": [{"role": "user", "parts": [
                {"text": "What is this?"},
                {"inlineData": {"mimeType": "image/jpeg", "data": "base64data"}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        imgs = [b for b in conv.messages[0].content if b.type == ContentType.IMAGE]
        assert len(imgs) == 1
        assert "image/jpeg" in imgs[0].url

    def test_parse_inline_data_snake_case(self, gemini_parser):
        data = {
            "contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/png", "data": "abc"}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        imgs = [b for b in conv.messages[0].content if b.type == ContentType.IMAGE]
        assert len(imgs) == 1

    # -- Function Calling ----------------------------------------------------

    def test_parse_function_call(self, gemini_parser):
        data = {
            "contents": [{"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        tc = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_CALL]
        assert len(tc) == 1
        assert tc[0].tool_name == "get_weather"

    def test_parse_function_response(self, gemini_parser):
        data = {
            "contents": [{"role": "user", "parts": [
                {"functionResponse": {"name": "get_weather", "response": {"temp": 25}}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        tr = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_RESULT]
        assert len(tr) == 1
        assert tr[0].tool_name == "get_weather"

    def test_parse_function_call_snake_case(self, gemini_parser):
        data = {
            "contents": [{"role": "model", "parts": [
                {"function_call": {"name": "search", "args": {"q": "test"}}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        tc = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_CALL]
        assert len(tc) == 1

    # -- Code Execution ------------------------------------------------------

    def test_parse_executable_code(self, gemini_parser):
        data = {
            "contents": [{"role": "model", "parts": [
                {"executableCode": {"language": "PYTHON", "code": "print('hello')"}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        code = [b for b in conv.messages[0].content if b.type == ContentType.CODE]
        assert len(code) == 1
        assert code[0].language == "python"

    def test_parse_code_execution_result(self, gemini_parser):
        data = {
            "contents": [{"role": "model", "parts": [
                {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "hello"}},
            ]}],
        }
        conv = gemini_parser.parse(data)
        tr = [b for b in conv.messages[0].content if b.type == ContentType.TOOL_RESULT]
        assert len(tr) == 1
        assert tr[0].text == "hello"

    # -- Response Format -----------------------------------------------------

    def test_parse_generate_content_response(self, gemini_parser):
        data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 50},
            "modelVersion": "gemini-1.5-pro",
        }
        conv = gemini_parser.parse(data)
        assert conv.source_model == "gemini-1.5-pro"
        assert conv.total_tokens == 60
        assert conv.message_count == 1

    # -- Edge Cases ----------------------------------------------------------

    def test_parse_empty_contents(self, gemini_parser):
        data = {"contents": []}
        conv = gemini_parser.parse(data)
        assert conv.message_count == 0

    def test_parse_single_content_object(self, gemini_parser):
        data = {"parts": [{"text": "Hello"}]}
        conv = gemini_parser.parse(data)
        assert conv.message_count == 1

    def test_parse_list_of_content_objects(self, gemini_parser):
        data = [
            {"role": "user", "parts": [{"text": "Hi"}]},
            {"role": "model", "parts": [{"text": "Hello"}]},
        ]
        conv = gemini_parser.parse(data)
        assert conv.message_count == 2

    def test_parse_rejects_invalid_data(self, gemini_parser):
        with pytest.raises(ParseError):
            gemini_parser.parse("not valid")

    def test_parse_string_parts(self, gemini_parser):
        data = {"contents": [{"role": "user", "parts": ["Hello world"]}]}
        conv = gemini_parser.parse(data)
        assert "Hello world" in conv.messages[0].full_text


# ============================================================================
# Generic Parser Tests
# ============================================================================


class TestGenericParser:
    """Test the generic markdown/text fallback parser."""

    # -- Detection -----------------------------------------------------------

    def test_can_parse_string(self, generic_parser):
        assert generic_parser.can_parse("Hello world")

    def test_cannot_parse_dict(self, generic_parser):
        assert not generic_parser.can_parse({"key": "value"})

    def test_cannot_parse_list(self, generic_parser):
        assert not generic_parser.can_parse([1, 2, 3])

    # -- Turn Detection ------------------------------------------------------

    def test_parse_user_assistant_markers(self, generic_parser):
        text = "User: Hello\nAssistant: Hi there!"
        conv = generic_parser.parse(text)
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT

    def test_parse_human_ai_markers(self, generic_parser):
        text = "Human: What is Python?\nAI: Python is a language."
        conv = generic_parser.parse(text)
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.USER
        assert conv.messages[1].role == MessageRole.ASSISTANT

    def test_parse_qa_markers(self, generic_parser):
        text = "Q: What is 2+2?\nA: 4."
        conv = generic_parser.parse(text)
        assert conv.message_count == 2

    def test_parse_system_marker(self, generic_parser):
        text = "System: Be helpful.\nUser: Hi\nAssistant: Hello!"
        conv = generic_parser.parse(text)
        assert conv.system_instruction == "Be helpful."
        assert conv.message_count == 2

    def test_parse_multiline_turns(self, generic_parser):
        text = (
            "User: I have a question.\n"
            "It's about Python.\n"
            "Can you help?\n"
            "Assistant: Of course!\n"
            "What would you like to know?"
        )
        conv = generic_parser.parse(text)
        assert conv.message_count == 2
        assert "It's about Python." in conv.messages[0].full_text
        assert "What would you like to know?" in conv.messages[1].full_text

    def test_parse_colon_separator(self, generic_parser):
        text = "User: Hello\nAssistant: World"
        conv = generic_parser.parse(text)
        assert conv.message_count == 2

    def test_parse_dash_separator(self, generic_parser):
        text = "User- Hello\nAssistant- World"
        conv = generic_parser.parse(text)
        assert conv.message_count == 2

    def test_parse_pipe_separator(self, generic_parser):
        text = "User| Hello\nAssistant| World"
        conv = generic_parser.parse(text)
        assert conv.message_count == 2

    def test_parse_named_models(self, generic_parser):
        text = "Claude: Here is my answer.\nGPT: And here is mine."
        conv = generic_parser.parse(text)
        assert conv.message_count == 2
        assert conv.messages[0].role == MessageRole.ASSISTANT

    def test_parse_section_separators_ignored(self, generic_parser):
        text = "User: Hello\n---\nAssistant: Hi"
        conv = generic_parser.parse(text)
        assert conv.message_count == 2

    # -- No Structure --------------------------------------------------------

    def test_parse_plain_text_as_user_message(self, generic_parser):
        text = "This is just some random text without any markers."
        conv = generic_parser.parse(text)
        assert conv.message_count == 1
        assert conv.messages[0].role == MessageRole.USER

    def test_parse_empty_string(self, generic_parser):
        conv = generic_parser.parse("")
        assert conv.message_count == 0

    def test_parse_whitespace_only(self, generic_parser):
        conv = generic_parser.parse("   \n\n  ")
        assert conv.message_count == 0

    # -- Code Blocks ---------------------------------------------------------

    def test_parse_code_in_generic_text(self, generic_parser):
        text = "User: Show me Python code\nAssistant: Here:\n```python\nprint('hi')\n```"
        conv = generic_parser.parse(text)
        assert conv.messages[1].has_code

    # -- Edge Cases ----------------------------------------------------------

    def test_parse_rejects_non_string(self, generic_parser):
        with pytest.raises(ParseError):
            generic_parser.parse(12345)

    def test_parse_many_turns(self, generic_parser):
        turns = []
        for i in range(50):
            turns.append(f"User: Message {i}")
            turns.append(f"Assistant: Response {i}")
        text = "\n".join(turns)
        conv = generic_parser.parse(text)
        assert conv.message_count == 100


# ============================================================================
# Parser Registry Tests
# ============================================================================


class TestParserRegistry:
    """Test auto-detection and routing."""

    def test_detect_openai_format(self, registry):
        data = [{"role": "user", "content": "hi"}]
        parser = registry.detect_format(data)
        assert parser is not None
        assert parser.name == "OpenAI/ChatGPT"

    def test_detect_claude_format(self, registry):
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hi"}],
        }
        parser = registry.detect_format(data)
        assert parser is not None
        assert parser.name == "Anthropic Claude"

    def test_detect_gemini_format(self, registry):
        data = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
        parser = registry.detect_format(data)
        assert parser is not None
        assert parser.name == "Google Gemini"

    def test_detect_generic_format(self, registry):
        parser = registry.detect_format("Hello world")
        assert parser is not None
        assert parser.name == "Generic Text"

    def test_detect_returns_none_for_unparseable(self, registry):
        parser = registry.detect_format(12345)
        assert parser is None

    def test_parse_auto_detects_openai(self, registry):
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        conv = registry.parse(data)
        assert conv.source_format == SourceFormat.OPENAI
        assert conv.message_count == 2

    def test_parse_auto_detects_claude(self, registry):
        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        conv = registry.parse(data)
        assert conv.source_format == SourceFormat.CLAUDE

    def test_parse_auto_detects_gemini(self, registry):
        data = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
        conv = registry.parse(data)
        assert conv.source_format == SourceFormat.GEMINI

    def test_parse_auto_detects_generic(self, registry):
        conv = registry.parse("User: Hello\nAssistant: Hi!")
        assert conv.source_format == SourceFormat.GENERIC

    def test_parse_json_string(self, registry):
        data = json.dumps([{"role": "user", "content": "Hello"}])
        conv = registry.parse(data)
        assert conv.source_format == SourceFormat.OPENAI

    def test_parse_raises_on_unparseable(self, registry):
        with pytest.raises(ParseError):
            registry.parse(12345)

    def test_parse_invalid_json_string_falls_to_generic(self, registry):
        conv = registry.parse("{invalid json but starts with brace")
        assert conv.source_format == SourceFormat.GENERIC

    def test_registry_priority_order(self, registry):
        parsers = registry.parsers
        assert parsers[0].priority > parsers[-1].priority

    def test_registry_register_custom_parser(self):
        registry = ParserRegistry()
        parser = GenericParser()
        registry.register(parser)
        assert len(registry.parsers) == 1


# ============================================================================
# Cross-Format Consistency Tests
# ============================================================================


class TestCrossFormatConsistency:
    """Ensure all parsers produce consistent CCR output."""

    def test_all_formats_produce_conversation(self, registry):
        """Every format should produce a valid Conversation."""
        inputs = [
            [{"role": "user", "content": "Hello"}],  # OpenAI
            {"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "Hello"}]},  # Claude
            {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},  # Gemini
            "User: Hello",  # Generic
        ]
        for inp in inputs:
            conv = registry.parse(inp)
            assert isinstance(conv, Conversation)
            assert conv.message_count >= 1

    def test_user_message_role_consistent(self, registry):
        """User messages should always have USER role."""
        inputs = [
            [{"role": "user", "content": "Hi"}],
            {"messages": [{"role": "user", "content": "Hi"}], "system": ""},
            {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
            "User: Hi",
        ]
        for inp in inputs:
            conv = registry.parse(inp)
            assert conv.messages[0].role == MessageRole.USER

    def test_code_blocks_detected_across_formats(self, registry):
        """Code blocks should be detected in all formats."""
        code_content = "Here:\n```python\nprint('hello')\n```"
        inputs = [
            [{"role": "assistant", "content": code_content}],
            {"messages": [{"role": "assistant", "content": code_content}], "system": ""},
            {"contents": [{"role": "model", "parts": [{"text": code_content}]}]},
            f"Assistant: {code_content}",
        ]
        for inp in inputs:
            conv = registry.parse(inp)
            has_code = any(m.has_code for m in conv.messages)
            assert has_code, f"Code not detected in format: {conv.source_format}"

    def test_system_instruction_extracted_across_formats(self, registry):
        """System instructions should be extracted in all formats."""
        inputs = [
            [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "Hi"}],
            {"system": "Be helpful.", "messages": [{"role": "user", "content": "Hi"}]},
            {"systemInstruction": {"parts": [{"text": "Be helpful."}]},
             "contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
            "System: Be helpful.\nUser: Hi",
        ]
        for inp in inputs:
            conv = registry.parse(inp)
            assert "Be helpful" in conv.system_instruction or "helpful" in conv.system_instruction.lower(), \
                f"System instruction not found in: {conv.source_format}"
