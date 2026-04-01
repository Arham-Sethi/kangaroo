"""Anthropic Claude conversation parser.

Handles the Claude API message format:

    {
        "model": "claude-sonnet-4-20250514",
        "system": "You are helpful.",    # or [{"type":"text","text":"..."}]
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hi!"},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {...}}
            ]}
        ]
    }

Claude's format differs from OpenAI in key ways:
    - Content is ALWAYS a list of typed blocks (text, image, tool_use, tool_result)
    - System prompt is a top-level field, not a message
    - Tool use has 'input' (dict) not 'arguments' (string)
    - Images use base64 'source' objects, not URLs
    - 'thinking' blocks contain chain-of-thought reasoning
    - No 'function_call' legacy format

Edge cases handled:
    - String content (simplified format, common in examples)
    - Thinking blocks (extended thinking / chain-of-thought)
    - Multi-part system prompts (list of text blocks)
    - Tool result blocks with is_error flag
    - Image blocks (base64 and URL sources)
    - Empty content arrays
"""

from __future__ import annotations

from typing import Any

from app.core.engine.ccr import (
    ContentBlock,
    ContentType,
    Conversation,
    Message,
    MessageRole,
    SourceFormat,
)
from app.core.engine.parsers.base import BaseParser, ParseError


class ClaudeParser(BaseParser):
    """Parser for Anthropic Claude API message format."""

    priority = 90

    @property
    def name(self) -> str:
        return "Anthropic Claude"

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.CLAUDE

    def can_parse(self, data: Any) -> bool:
        """Detect Claude format by structural markers.

        Claude format indicators:
            - dict with 'messages' list AND ('model' containing 'claude' OR 'system' key)
            - dict with 'content' as list containing typed blocks with 'type' key
            - dict with 'role' and 'content' list containing {'type': 'text'} blocks
        """
        if not isinstance(data, dict):
            return False

        # Full Claude API request/response
        if "messages" in data:
            messages = data.get("messages")
            if not isinstance(messages, list):
                return False
            model = self._safe_str(data.get("model", "")).lower()
            if "claude" in model:
                return True
            # Has system as a top-level key (Claude-specific)
            if "system" in data:
                return True
            # Check if messages have Claude-style content blocks
            if messages and isinstance(messages[0], dict):
                content = messages[0].get("content")
                if isinstance(content, list) and content:
                    first_block = content[0]
                    if isinstance(first_block, dict) and "type" in first_block:
                        return True

        # Claude API response wrapper
        if "type" in data and data.get("type") == "message":
            return True

        # Single Claude-format message with content blocks
        if "role" in data and "content" in data:
            content = data.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "type" in first:
                    block_type = first.get("type", "")
                    if block_type in ("text", "tool_use", "tool_result", "thinking"):
                        return True

        return False

    def parse(self, data: Any) -> Conversation:
        """Parse Claude API data into a normalized Conversation."""
        if not isinstance(data, dict):
            raise ParseError(
                "Expected a dict with Claude API format.",
                parser_name=self.name,
                source_format=self.source_format,
            )

        # Claude API response
        if data.get("type") == "message":
            return self._parse_response(data)

        # Claude API request (messages + optional system)
        if "messages" in data:
            return self._parse_request(data)

        # Single message
        if "role" in data and "content" in data:
            return self._parse_request({"messages": [data]})

        raise ParseError(
            "Unrecognized Claude format. Expected messages array or API response.",
            parser_name=self.name,
            source_format=self.source_format,
        )

    def _parse_request(self, data: dict) -> Conversation:
        """Parse a Claude API request (messages + system prompt)."""
        # Extract system instruction
        system_instruction = self._extract_system(data.get("system"))

        # Parse messages
        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ParseError(
                "'messages' must be a list.",
                parser_name=self.name,
                source_format=self.source_format,
            )

        parsed_messages: list[Message] = []
        model = self._safe_str(data.get("model", ""))

        for raw_msg in raw_messages:
            if not isinstance(raw_msg, dict):
                continue

            msg = self._parse_message(raw_msg, model)
            if msg is not None:
                parsed_messages.append(msg)

        return Conversation(
            source_format=SourceFormat.CLAUDE,
            source_llm="anthropic",
            source_model=model,
            system_instruction=system_instruction,
            messages=tuple(parsed_messages),
            message_count=len(parsed_messages),
        )

    def _parse_response(self, data: dict) -> Conversation:
        """Parse a Claude API response object.

        {
            "type": "message",
            "role": "assistant",
            "content": [...],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 200}
        }
        """
        content_blocks = self._parse_content_blocks(data.get("content", []))
        model = self._safe_str(data.get("model", ""))

        # Extract token counts
        usage = data.get("usage", {})
        total_tokens = 0
        if isinstance(usage, dict):
            total_tokens = (
                self._safe_int(usage.get("input_tokens", 0))
                + self._safe_int(usage.get("output_tokens", 0))
            )

        msg = Message(
            role=MessageRole.ASSISTANT,
            content=tuple(content_blocks),
            model=model,
            finish_reason=self._safe_str(data.get("stop_reason", "")),
            message_id=self._safe_str(data.get("id", "")),
        )

        return Conversation(
            source_format=SourceFormat.CLAUDE,
            source_llm="anthropic",
            source_model=model,
            messages=(msg,),
            total_tokens=total_tokens,
            message_count=1,
            raw_metadata={
                "stop_reason": self._safe_str(data.get("stop_reason", "")),
                "stop_sequence": data.get("stop_sequence"),
            },
        )

    def _parse_message(self, raw_msg: dict, model: str = "") -> Message | None:
        """Parse a single Claude message into our Message format."""
        role_str = self._safe_str(raw_msg.get("role", "")).lower()
        if not role_str:
            return None

        role = self._map_role(role_str)
        content = raw_msg.get("content")

        # Claude content can be string (simplified) or list of blocks
        if isinstance(content, str):
            if not content.strip():
                return None
            blocks = self._extract_blocks_from_text(content)
        elif isinstance(content, list):
            blocks = self._parse_content_blocks(content)
        else:
            return None

        if not blocks:
            return None

        return Message(
            role=role,
            content=tuple(blocks),
            model=model,
            name=self._safe_str(raw_msg.get("name", "")),
        )

    def _parse_content_blocks(self, content: list) -> list[ContentBlock]:
        """Parse Claude's typed content block array.

        Block types:
            - text: Plain text content
            - tool_use: Tool/function call (id, name, input)
            - tool_result: Tool/function result (tool_use_id, content)
            - image: Image (base64 or URL source)
            - thinking: Chain-of-thought reasoning (extended thinking)
        """
        blocks: list[ContentBlock] = []

        for block in content:
            if isinstance(block, str):
                if block.strip():
                    blocks.extend(self._extract_blocks_from_text(block))
                continue

            if not isinstance(block, dict):
                continue

            block_type = self._safe_str(block.get("type", ""))

            if block_type == "text":
                text = self._safe_str(block.get("text", ""))
                if text.strip():
                    blocks.extend(self._extract_blocks_from_text(text))

            elif block_type == "tool_use":
                input_data = block.get("input", {})
                import json
                input_str = json.dumps(input_data) if isinstance(input_data, dict) else self._safe_str(input_data)
                blocks.append(ContentBlock(
                    type=ContentType.TOOL_CALL,
                    text=input_str,
                    tool_name=self._safe_str(block.get("name", "")),
                    tool_call_id=self._safe_str(block.get("id", "")),
                ))

            elif block_type == "tool_result":
                result_content = block.get("content", "")
                result_text = ""
                if isinstance(result_content, str):
                    result_text = result_content
                elif isinstance(result_content, list):
                    # Tool result can contain nested content blocks
                    parts = []
                    for part in result_content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(self._safe_str(part.get("text", "")))
                    result_text = "\n".join(parts)

                is_error = block.get("is_error", False)
                blocks.append(ContentBlock(
                    type=ContentType.TOOL_RESULT if not is_error else ContentType.ERROR,
                    text=result_text,
                    tool_call_id=self._safe_str(block.get("tool_use_id", "")),
                    metadata={"is_error": is_error},
                ))

            elif block_type == "image":
                source = block.get("source", {})
                url = ""
                if isinstance(source, dict):
                    if source.get("type") == "url":
                        url = self._safe_str(source.get("url", ""))
                    elif source.get("type") == "base64":
                        media_type = self._safe_str(source.get("media_type", ""))
                        url = f"data:{media_type};base64,[embedded]"
                blocks.append(ContentBlock(
                    type=ContentType.IMAGE,
                    url=url,
                    alt_text="Image in conversation",
                ))

            elif block_type == "thinking":
                thinking_text = self._safe_str(block.get("thinking", ""))
                if thinking_text.strip():
                    blocks.append(ContentBlock(
                        type=ContentType.THINKING,
                        text=thinking_text,
                    ))

        return blocks

    def _extract_system(self, system: Any) -> str:
        """Extract system instruction from Claude's system field.

        System can be:
            - A string: "You are helpful."
            - A list of text blocks: [{"type":"text","text":"You are helpful."}]
        """
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            parts: list[str] = []
            for block in system:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(self._safe_str(block.get("text", "")))
            return "\n".join(parts)
        return ""

    def _map_role(self, role_str: str) -> MessageRole:
        """Map Claude role strings to our MessageRole enum."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        return role_map.get(role_str.lower(), MessageRole.USER)

    def _extract_blocks_from_text(self, text: str) -> list[ContentBlock]:
        """Extract code blocks from text, same as OpenAI parser."""
        blocks: list[ContentBlock] = []
        lines = text.split("\n")
        current_text: list[str] = []
        in_code_block = False
        code_language = ""
        code_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```") and not in_code_block:
                if current_text:
                    joined = "\n".join(current_text).strip()
                    if joined:
                        blocks.append(ContentBlock(type=ContentType.TEXT, text=joined))
                    current_text = []
                in_code_block = True
                code_language = stripped[3:].strip().split()[0] if len(stripped) > 3 else ""
                code_lines = []

            elif stripped.startswith("```") and in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                if code_text.strip():
                    blocks.append(ContentBlock(
                        type=ContentType.CODE,
                        text=code_text,
                        language=code_language,
                    ))
                code_lines = []
                code_language = ""

            elif in_code_block:
                code_lines.append(line)

            else:
                current_text.append(line)

        if in_code_block and code_lines:
            blocks.append(ContentBlock(
                type=ContentType.CODE,
                text="\n".join(code_lines),
                language=code_language,
            ))
        elif current_text:
            joined = "\n".join(current_text).strip()
            if joined:
                blocks.append(ContentBlock(type=ContentType.TEXT, text=joined))

        return blocks if blocks else [ContentBlock(type=ContentType.TEXT, text=text)]
