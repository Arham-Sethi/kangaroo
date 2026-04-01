"""OpenAI / ChatGPT conversation parser.

Handles TWO distinct formats:

1. **OpenAI API format** — The standard message array used by the API:
    [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

2. **ChatGPT export format** — The JSON export from ChatGPT's UI:
    {
        "title": "My Conversation",
        "mapping": {
            "node-id": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hello"]},
                    "create_time": 1234567890
                },
                "parent": "parent-node-id",
                "children": ["child-node-id"]
            }
        }
    }

The export format uses a tree structure (for branching conversations).
We linearize it by following the main branch (last child at each fork).

Edge cases handled:
    - Multi-modal messages (text + image_url)
    - Tool/function calls (both legacy function_call and modern tool_calls)
    - Code interpreter outputs
    - System messages (extracted as system_instruction)
    - Empty messages (skipped)
    - Malformed data (graceful degradation with warnings)
"""

from __future__ import annotations

from datetime import datetime, timezone
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


class OpenAIParser(BaseParser):
    """Parser for OpenAI API messages and ChatGPT export JSON."""

    priority = 100  # Highest — most common format

    @property
    def name(self) -> str:
        return "OpenAI/ChatGPT"

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.OPENAI

    def can_parse(self, data: Any) -> bool:
        """Detect OpenAI format by structural markers.

        API format: list of dicts with 'role' key
        Export format: dict with 'mapping' key
        """
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            return isinstance(first, dict) and "role" in first

        if isinstance(data, dict):
            # ChatGPT export has 'mapping' key
            if "mapping" in data:
                return True
            # Single message object
            if "role" in data and "content" in data:
                return True
            # API response wrapper
            if "choices" in data:
                return True

        return False

    def parse(self, data: Any) -> Conversation:
        """Parse OpenAI data into a normalized Conversation."""
        if isinstance(data, dict):
            if "mapping" in data:
                return self._parse_export(data)
            if "choices" in data:
                return self._parse_api_response(data)
            if "role" in data:
                return self._parse_message_array([data])

        if isinstance(data, list):
            return self._parse_message_array(data)

        raise ParseError(
            "Expected a list of messages, ChatGPT export JSON, or API response.",
            parser_name=self.name,
            source_format=self.source_format,
        )

    # -- API Message Array ---------------------------------------------------

    def _parse_message_array(self, messages: list[dict]) -> Conversation:
        """Parse the standard OpenAI API message array format."""
        system_instruction = ""
        parsed_messages: list[Message] = []
        total_tokens = 0
        source_model = ""

        for raw_msg in messages:
            if not isinstance(raw_msg, dict):
                continue

            role_str = self._safe_str(raw_msg.get("role", "")).lower()
            if not role_str:
                continue

            # Extract system instruction
            if role_str == "system":
                content = raw_msg.get("content", "")
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, list):
                    system_instruction = self._extract_text_from_content_list(content)
                continue

            # Map role
            role = self._map_role(role_str)

            # Parse content blocks
            content_blocks = self._parse_content(raw_msg)

            # Parse tool calls (modern format)
            tool_call_blocks = self._parse_tool_calls(raw_msg)
            all_blocks = content_blocks + tool_call_blocks

            if not all_blocks:
                continue

            # Extract model info
            if raw_msg.get("model"):
                source_model = raw_msg["model"]

            msg = Message(
                role=role,
                content=tuple(all_blocks),
                name=self._safe_str(raw_msg.get("name", "")),
                model=source_model,
                finish_reason=self._safe_str(raw_msg.get("finish_reason", "")),
                message_id=self._safe_str(raw_msg.get("id", "")),
                metadata={
                    k: v for k, v in raw_msg.items()
                    if k not in ("role", "content", "name", "tool_calls",
                                 "function_call", "model", "id")
                    and v is not None
                },
            )
            parsed_messages.append(msg)

        return Conversation(
            source_format=SourceFormat.OPENAI,
            source_llm="openai",
            source_model=source_model,
            system_instruction=system_instruction,
            messages=tuple(parsed_messages),
            total_tokens=total_tokens,
            message_count=len(parsed_messages),
        )

    def _parse_content(self, raw_msg: dict) -> list[ContentBlock]:
        """Parse the 'content' field of an OpenAI message.

        Content can be:
            - A string: simple text message
            - A list: multi-modal content (text + images + etc.)
            - None: message with only tool_calls
        """
        content = raw_msg.get("content")

        if content is None:
            return []

        if isinstance(content, str):
            if not content.strip():
                return []
            return self._extract_blocks_from_text(content)

        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for part in content:
                if isinstance(part, str):
                    if part.strip():
                        blocks.extend(self._extract_blocks_from_text(part))
                elif isinstance(part, dict):
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        text = self._safe_str(part.get("text", ""))
                        if text.strip():
                            blocks.extend(self._extract_blocks_from_text(text))
                    elif part_type == "image_url":
                        image_data = part.get("image_url", {})
                        url = ""
                        if isinstance(image_data, dict):
                            url = self._safe_str(image_data.get("url", ""))
                        elif isinstance(image_data, str):
                            url = image_data
                        blocks.append(ContentBlock(
                            type=ContentType.IMAGE,
                            url=url,
                            alt_text="User provided image",
                        ))
                    elif part_type == "input_audio":
                        blocks.append(ContentBlock(
                            type=ContentType.TEXT,
                            text="[Audio input]",
                            metadata={"original_type": "input_audio"},
                        ))
            return blocks

        return []

    def _parse_tool_calls(self, raw_msg: dict) -> list[ContentBlock]:
        """Parse tool_calls and legacy function_call from OpenAI messages."""
        blocks: list[ContentBlock] = []

        # Modern tool_calls format
        tool_calls = raw_msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    continue
                blocks.append(ContentBlock(
                    type=ContentType.TOOL_CALL,
                    text=self._safe_str(func.get("arguments", "{}")),
                    tool_name=self._safe_str(func.get("name", "")),
                    tool_call_id=self._safe_str(tc.get("id", "")),
                ))

        # Legacy function_call format
        func_call = raw_msg.get("function_call")
        if isinstance(func_call, dict):
            blocks.append(ContentBlock(
                type=ContentType.TOOL_CALL,
                text=self._safe_str(func_call.get("arguments", "{}")),
                tool_name=self._safe_str(func_call.get("name", "")),
            ))

        # Tool result (role=tool message)
        role = self._safe_str(raw_msg.get("role", ""))
        if role == "tool":
            content = self._safe_str(raw_msg.get("content", ""))
            blocks.append(ContentBlock(
                type=ContentType.TOOL_RESULT,
                text=content,
                tool_name=self._safe_str(raw_msg.get("name", "")),
                tool_call_id=self._safe_str(raw_msg.get("tool_call_id", "")),
            ))

        return blocks

    # -- ChatGPT Export Format -----------------------------------------------

    def _parse_export(self, data: dict) -> Conversation:
        """Parse ChatGPT's exported JSON conversation tree.

        ChatGPT stores conversations as a tree (for branching). We
        linearize by following each node's last child (the main branch).
        """
        title = self._safe_str(data.get("title", ""))
        mapping = data.get("mapping", {})
        model_slug = self._safe_str(data.get("default_model_slug", ""))
        create_time = data.get("create_time")

        if not isinstance(mapping, dict) or not mapping:
            raise ParseError(
                "ChatGPT export has empty or invalid 'mapping'.",
                parser_name=self.name,
                source_format=SourceFormat.OPENAI_EXPORT,
            )

        # Find the root node (no parent or parent not in mapping)
        root_id = None
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent is None or parent not in mapping:
                root_id = node_id
                break

        if root_id is None:
            raise ParseError(
                "Cannot find root node in ChatGPT export tree.",
                parser_name=self.name,
                source_format=SourceFormat.OPENAI_EXPORT,
            )

        # Linearize the tree by following the main branch
        ordered_messages = self._linearize_tree(mapping, root_id)

        # Parse each node into a Message
        system_instruction = ""
        parsed_messages: list[Message] = []

        for node in ordered_messages:
            msg_data = node.get("message")
            if not isinstance(msg_data, dict):
                continue

            author = msg_data.get("author", {})
            if not isinstance(author, dict):
                continue

            role_str = self._safe_str(author.get("role", "")).lower()
            if not role_str or role_str == "system":
                # Extract system content
                content_obj = msg_data.get("content", {})
                if isinstance(content_obj, dict):
                    parts = content_obj.get("parts", [])
                    if isinstance(parts, list):
                        text = " ".join(
                            self._safe_str(p) for p in parts
                            if isinstance(p, str) and p.strip()
                        )
                        if text.strip():
                            system_instruction = text
                continue

            role = self._map_role(role_str)

            # Extract content from ChatGPT's nested structure
            content_obj = msg_data.get("content", {})
            content_blocks = self._parse_export_content(content_obj)

            if not content_blocks:
                continue

            # Timestamp
            create_ts = msg_data.get("create_time")
            timestamp = None
            if isinstance(create_ts, (int, float)) and create_ts > 0:
                timestamp = datetime.fromtimestamp(create_ts, tz=timezone.utc)

            # Model
            msg_metadata = msg_data.get("metadata", {})
            msg_model = ""
            if isinstance(msg_metadata, dict):
                msg_model = self._safe_str(msg_metadata.get("model_slug", model_slug))

            msg = Message(
                role=role,
                content=tuple(content_blocks),
                model=msg_model,
                timestamp=timestamp,
                message_id=self._safe_str(msg_data.get("id", "")),
            )
            parsed_messages.append(msg)

        created_at = datetime.now(timezone.utc)
        if isinstance(create_time, (int, float)) and create_time > 0:
            created_at = datetime.fromtimestamp(create_time, tz=timezone.utc)

        return Conversation(
            source_format=SourceFormat.OPENAI_EXPORT,
            source_llm="openai",
            source_model=model_slug,
            title=title,
            system_instruction=system_instruction,
            messages=tuple(parsed_messages),
            message_count=len(parsed_messages),
            created_at=created_at,
            raw_metadata={
                "conversation_id": self._safe_str(data.get("conversation_id", "")),
                "is_archived": data.get("is_archived", False),
            },
        )

    def _linearize_tree(self, mapping: dict, root_id: str) -> list[dict]:
        """Walk the conversation tree and produce a linear message list.

        At each node, we follow the LAST child (most recent branch).
        This produces the main conversation thread.
        """
        result: list[dict] = []
        current_id = root_id
        visited: set[str] = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            node = mapping.get(current_id)
            if node is None:
                break

            result.append(node)

            children = node.get("children", [])
            if isinstance(children, list) and children:
                current_id = children[-1]  # Follow last child
            else:
                break

        return result

    def _parse_export_content(self, content_obj: Any) -> list[ContentBlock]:
        """Parse ChatGPT export's nested content structure."""
        if isinstance(content_obj, str):
            if content_obj.strip():
                return self._extract_blocks_from_text(content_obj)
            return []

        if not isinstance(content_obj, dict):
            return []

        content_type = content_obj.get("content_type", "text")
        parts = content_obj.get("parts", [])

        if not isinstance(parts, list):
            return []

        blocks: list[ContentBlock] = []
        for part in parts:
            if isinstance(part, str) and part.strip():
                blocks.extend(self._extract_blocks_from_text(part))
            elif isinstance(part, dict):
                # Image or asset reference
                asset_pointer = part.get("asset_pointer", "")
                if asset_pointer:
                    blocks.append(ContentBlock(
                        type=ContentType.IMAGE,
                        url=self._safe_str(asset_pointer),
                        alt_text="ChatGPT generated image",
                    ))

        if content_type == "code" and not blocks:
            text = content_obj.get("text", "")
            if isinstance(text, str) and text.strip():
                lang = self._safe_str(content_obj.get("language", ""))
                blocks.append(ContentBlock(
                    type=ContentType.CODE,
                    text=text,
                    language=lang,
                ))

        return blocks

    # -- API Response Wrapper ------------------------------------------------

    def _parse_api_response(self, data: dict) -> Conversation:
        """Parse an OpenAI API completion response.

        {
            "choices": [{"message": {"role": "assistant", "content": "..."}}],
            "model": "gpt-4o",
            "usage": {"total_tokens": 1234}
        }
        """
        choices = data.get("choices", [])
        messages: list[dict] = []

        for choice in choices:
            if isinstance(choice, dict):
                msg = choice.get("message") or choice.get("delta")
                if isinstance(msg, dict):
                    messages.append(msg)

        conv = self._parse_message_array(messages)

        # Enrich with response metadata
        model = self._safe_str(data.get("model", ""))
        usage = data.get("usage", {})
        total_tokens = 0
        if isinstance(usage, dict):
            total_tokens = self._safe_int(usage.get("total_tokens", 0))

        return Conversation(
            source_format=conv.source_format,
            source_llm=conv.source_llm,
            source_model=model or conv.source_model,
            system_instruction=conv.system_instruction,
            messages=conv.messages,
            total_tokens=total_tokens,
            message_count=conv.message_count,
            raw_metadata={"response_id": self._safe_str(data.get("id", ""))},
        )

    # -- Utilities -----------------------------------------------------------

    def _map_role(self, role_str: str) -> MessageRole:
        """Map OpenAI role strings to our MessageRole enum."""
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.TOOL,
            "function": MessageRole.TOOL,
        }
        return role_map.get(role_str.lower(), MessageRole.USER)

    def _extract_text_from_content_list(self, content: list) -> str:
        """Extract text from a content list (multi-modal format)."""
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(self._safe_str(part.get("text", "")))
        return "\n".join(texts)

    def _extract_blocks_from_text(self, text: str) -> list[ContentBlock]:
        """Extract code blocks from text, producing alternating text/code blocks.

        Parses markdown-style fenced code blocks:
            ```python
            def foo():
                pass
            ```

        Text outside code blocks becomes TEXT blocks.
        Code inside fences becomes CODE blocks with language detection.
        """
        blocks: list[ContentBlock] = []
        lines = text.split("\n")
        current_text: list[str] = []
        in_code_block = False
        code_language = ""
        code_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```") and not in_code_block:
                # Flush accumulated text
                if current_text:
                    joined = "\n".join(current_text).strip()
                    if joined:
                        blocks.append(ContentBlock(type=ContentType.TEXT, text=joined))
                    current_text = []

                # Start code block
                in_code_block = True
                code_language = stripped[3:].strip().split()[0] if len(stripped) > 3 else ""
                code_lines = []

            elif stripped.startswith("```") and in_code_block:
                # End code block
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

        # Handle unclosed code block
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
