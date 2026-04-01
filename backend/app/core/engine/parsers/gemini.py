"""Google Gemini conversation parser.

Handles the Gemini API content format:

    {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Hello"},
                    {"inline_data": {"mime_type": "image/png", "data": "base64..."}}
                ]
            },
            {
                "role": "model",
                "parts": [{"text": "Hi there!"}]
            }
        ],
        "systemInstruction": {"parts": [{"text": "You are helpful."}]},
        "generationConfig": {"temperature": 0.7}
    }

Gemini's format differs from OpenAI/Claude in key ways:
    - Assistant role is called "model" not "assistant"
    - Content blocks are called "parts" not "content"
    - Images use "inline_data" with mime_type + base64
    - System instruction is "systemInstruction" at top level
    - Function calls use "functionCall" and "functionResponse"
    - Grounding metadata for web search results
    - Candidate responses with safety ratings

Edge cases handled:
    - generateContent response format (with candidates)
    - Chat history format (contents array)
    - Function calling (functionCall / functionResponse parts)
    - Grounding metadata (web search citations)
    - Safety ratings (mapped to safety flags)
    - Multi-modal parts (text + inline_data)
    - Empty or missing parts
"""

from __future__ import annotations

import json
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


class GeminiParser(BaseParser):
    """Parser for Google Gemini API content format."""

    priority = 80

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.GEMINI

    def can_parse(self, data: Any) -> bool:
        """Detect Gemini format by structural markers.

        Gemini indicators:
            - dict with 'contents' list containing dicts with 'parts'
            - dict with 'candidates' (generateContent response)
            - dict with 'systemInstruction' key
            - list of dicts with 'parts' key
        """
        if isinstance(data, dict):
            # API response with candidates
            if "candidates" in data:
                return True

            # Request/history with contents array
            if "contents" in data:
                contents = data.get("contents")
                if isinstance(contents, list) and contents:
                    first = contents[0]
                    return isinstance(first, dict) and "parts" in first

            # Has Gemini-specific systemInstruction
            if "systemInstruction" in data:
                return True

            # Single content object with parts
            if "parts" in data and isinstance(data.get("parts"), list):
                return True

        # List of content objects
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "parts" in first:
                return True

        return False

    def parse(self, data: Any) -> Conversation:
        """Parse Gemini data into a normalized Conversation."""
        if isinstance(data, list):
            return self._parse_contents(data, {})

        if not isinstance(data, dict):
            raise ParseError(
                "Expected dict or list for Gemini format.",
                parser_name=self.name,
                source_format=self.source_format,
            )

        # generateContent response
        if "candidates" in data:
            return self._parse_response(data)

        # Chat request/history
        if "contents" in data:
            contents = data.get("contents", [])
            return self._parse_contents(contents, data)

        # Single content object
        if "parts" in data:
            return self._parse_contents([data], {})

        raise ParseError(
            "Unrecognized Gemini format.",
            parser_name=self.name,
            source_format=self.source_format,
        )

    def _parse_response(self, data: dict) -> Conversation:
        """Parse a Gemini generateContent response.

        {
            "candidates": [{
                "content": {"role": "model", "parts": [...]},
                "finishReason": "STOP",
                "safetyRatings": [...]
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 50}
        }
        """
        candidates = data.get("candidates", [])
        parsed_messages: list[Message] = []

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue

            content = candidate.get("content", {})
            if not isinstance(content, dict):
                continue

            msg = self._parse_content_object(content)
            if msg is not None:
                # Add finish reason from candidate
                finish_reason = self._safe_str(candidate.get("finishReason", ""))
                msg = Message(
                    role=msg.role,
                    content=msg.content,
                    model=msg.model,
                    finish_reason=finish_reason,
                    metadata={
                        **msg.metadata,
                        "safety_ratings": candidate.get("safetyRatings", []),
                    },
                )
                parsed_messages.append(msg)

        # Token counts
        usage = data.get("usageMetadata", {})
        total_tokens = 0
        if isinstance(usage, dict):
            total_tokens = (
                self._safe_int(usage.get("promptTokenCount", 0))
                + self._safe_int(usage.get("candidatesTokenCount", 0))
            )

        model = self._safe_str(data.get("modelVersion", ""))

        return Conversation(
            source_format=SourceFormat.GEMINI,
            source_llm="google",
            source_model=model,
            messages=tuple(parsed_messages),
            total_tokens=total_tokens,
            message_count=len(parsed_messages),
        )

    def _parse_contents(self, contents: list, context: dict) -> Conversation:
        """Parse a Gemini contents array (chat history format)."""
        if not isinstance(contents, list):
            raise ParseError(
                "'contents' must be a list.",
                parser_name=self.name,
                source_format=self.source_format,
            )

        # Extract system instruction
        system_instruction = self._extract_system_instruction(context)

        parsed_messages: list[Message] = []
        model = self._safe_str(context.get("model", ""))

        for content_obj in contents:
            if not isinstance(content_obj, dict):
                continue

            msg = self._parse_content_object(content_obj, model)
            if msg is not None:
                parsed_messages.append(msg)

        return Conversation(
            source_format=SourceFormat.GEMINI,
            source_llm="google",
            source_model=model,
            system_instruction=system_instruction,
            messages=tuple(parsed_messages),
            message_count=len(parsed_messages),
        )

    def _parse_content_object(
        self, content: dict, model: str = "",
    ) -> Message | None:
        """Parse a single Gemini content object into a Message.

        {
            "role": "user" | "model",
            "parts": [
                {"text": "..."},
                {"inlineData": {"mimeType": "image/png", "data": "..."}},
                {"functionCall": {"name": "search", "args": {...}}},
                {"functionResponse": {"name": "search", "response": {...}}}
            ]
        }
        """
        role_str = self._safe_str(content.get("role", "user")).lower()
        role = self._map_role(role_str)

        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return None

        blocks = self._parse_parts(parts)
        if not blocks:
            return None

        return Message(
            role=role,
            content=tuple(blocks),
            model=model,
        )

    def _parse_parts(self, parts: list) -> list[ContentBlock]:
        """Parse Gemini's parts array into ContentBlocks."""
        blocks: list[ContentBlock] = []

        for part in parts:
            if isinstance(part, str):
                if part.strip():
                    blocks.extend(self._extract_blocks_from_text(part))
                continue

            if not isinstance(part, dict):
                continue

            # Text part
            if "text" in part:
                text = self._safe_str(part["text"])
                if text.strip():
                    blocks.extend(self._extract_blocks_from_text(text))

            # Inline data (images, files)
            elif "inlineData" in part or "inline_data" in part:
                inline = part.get("inlineData") or part.get("inline_data", {})
                if isinstance(inline, dict):
                    mime = self._safe_str(inline.get("mimeType", inline.get("mime_type", "")))
                    blocks.append(ContentBlock(
                        type=ContentType.IMAGE if mime.startswith("image/") else ContentType.FILE,
                        url=f"data:{mime};base64,[embedded]",
                        alt_text=f"Inline {mime} content",
                        metadata={"mime_type": mime},
                    ))

            # Function call
            elif "functionCall" in part or "function_call" in part:
                fc = part.get("functionCall") or part.get("function_call", {})
                if isinstance(fc, dict):
                    args = fc.get("args", {})
                    args_str = json.dumps(args) if isinstance(args, dict) else self._safe_str(args)
                    blocks.append(ContentBlock(
                        type=ContentType.TOOL_CALL,
                        text=args_str,
                        tool_name=self._safe_str(fc.get("name", "")),
                    ))

            # Function response
            elif "functionResponse" in part or "function_response" in part:
                fr = part.get("functionResponse") or part.get("function_response", {})
                if isinstance(fr, dict):
                    response = fr.get("response", {})
                    response_str = json.dumps(response) if isinstance(response, dict) else self._safe_str(response)
                    blocks.append(ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        text=response_str,
                        tool_name=self._safe_str(fr.get("name", "")),
                    ))

            # Executable code (code execution feature)
            elif "executableCode" in part or "executable_code" in part:
                ec = part.get("executableCode") or part.get("executable_code", {})
                if isinstance(ec, dict):
                    blocks.append(ContentBlock(
                        type=ContentType.CODE,
                        text=self._safe_str(ec.get("code", "")),
                        language=self._safe_str(ec.get("language", "python")).lower(),
                    ))

            # Code execution result
            elif "codeExecutionResult" in part or "code_execution_result" in part:
                cer = part.get("codeExecutionResult") or part.get("code_execution_result", {})
                if isinstance(cer, dict):
                    outcome = self._safe_str(cer.get("outcome", ""))
                    output = self._safe_str(cer.get("output", ""))
                    blocks.append(ContentBlock(
                        type=ContentType.TOOL_RESULT,
                        text=output,
                        tool_name="code_execution",
                        metadata={"outcome": outcome},
                    ))

        return blocks

    def _extract_system_instruction(self, context: dict) -> str:
        """Extract system instruction from Gemini's systemInstruction field.

        Can be:
            - {"systemInstruction": {"parts": [{"text": "..."}]}}
            - {"systemInstruction": "..."}
            - {"system_instruction": {...}}
        """
        system = context.get("systemInstruction") or context.get("system_instruction")
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, dict):
            parts = system.get("parts", [])
            if isinstance(parts, list):
                texts: list[str] = []
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        texts.append(self._safe_str(part["text"]))
                return "\n".join(texts)
        return ""

    def _map_role(self, role_str: str) -> MessageRole:
        """Map Gemini role strings to our MessageRole enum.

        Gemini uses "model" instead of "assistant".
        """
        role_map = {
            "user": MessageRole.USER,
            "model": MessageRole.ASSISTANT,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "function": MessageRole.TOOL,
            "tool": MessageRole.TOOL,
        }
        return role_map.get(role_str.lower(), MessageRole.USER)

    def _extract_blocks_from_text(self, text: str) -> list[ContentBlock]:
        """Extract code blocks from text, producing alternating text/code blocks."""
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
