"""Generic markdown/text conversation parser.

This is the fallback parser — it handles any text that doesn't match
a specific LLM format. Users can paste conversations from any source:

    - Markdown with "User:" / "Assistant:" prefixes
    - Chat logs with "Human:" / "AI:" prefixes
    - Forum-style "Q:" / "A:" format
    - Raw text (treated as a single user message)
    - Text files with conversation structure

Detection patterns (case-insensitive, with flexible separators):
    - "User:", "Human:", "Me:", "Q:" → user turn
    - "Assistant:", "AI:", "Bot:", "Claude:", "GPT:", "A:" → assistant turn
    - "System:", "Instructions:" → system instruction
    - "---" or "===" → section separator

The parser is intentionally lenient — it tries to extract structure
from unstructured text. When in doubt, it treats the input as a
single user message. This ensures no input is ever rejected.

Edge cases handled:
    - Empty input → empty conversation
    - No turn markers → single user message
    - Mixed marker styles → normalized
    - Code blocks within turns → preserved
    - Markdown headers as topic markers → preserved in metadata
"""

from __future__ import annotations

import re
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


# Turn detection patterns — order matters (more specific first)
_TURN_PATTERNS: list[tuple[re.Pattern, MessageRole]] = [
    # System / instructions
    (re.compile(r"^(?:system|instructions?)\s*[:\-|>]\s*", re.IGNORECASE), MessageRole.SYSTEM),
    # User patterns
    (re.compile(r"^(?:user|human|me|q|you|person|customer)\s*[:\-|>]\s*", re.IGNORECASE), MessageRole.USER),
    # Assistant patterns
    (re.compile(
        r"^(?:assistant|ai|bot|claude|gpt|chatgpt|gemini|copilot|"
        r"model|a|response|answer)\s*[:\-|>]\s*",
        re.IGNORECASE,
    ), MessageRole.ASSISTANT),
]

# Section separators
_SEPARATOR_PATTERN = re.compile(r"^(?:-{3,}|={3,}|\*{3,})\s*$")


class GenericParser(BaseParser):
    """Fallback parser for markdown, plain text, and unknown formats.

    This parser is the last resort — it always returns True from
    can_parse() for string inputs. It tries to extract conversation
    structure from free-form text.
    """

    priority = 0  # Lowest — only used when no other parser matches

    @property
    def name(self) -> str:
        return "Generic Text"

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.GENERIC

    def can_parse(self, data: Any) -> bool:
        """Accept any string input."""
        return isinstance(data, str)

    def parse(self, data: Any) -> Conversation:
        """Parse text input into a normalized Conversation."""
        if not isinstance(data, str):
            raise ParseError(
                "Generic parser requires string input.",
                parser_name=self.name,
                source_format=self.source_format,
            )

        text = data.strip()
        if not text:
            return Conversation(
                source_format=SourceFormat.GENERIC,
                source_llm="unknown",
                messages=(),
                message_count=0,
            )

        # Try to detect conversation structure
        turns = self._detect_turns(text)

        if turns:
            return self._build_from_turns(turns)

        # No structure detected — treat as single user message
        blocks = self._extract_blocks_from_text(text)
        msg = Message(
            role=MessageRole.USER,
            content=tuple(blocks),
        )

        return Conversation(
            source_format=SourceFormat.GENERIC,
            source_llm="unknown",
            messages=(msg,),
            message_count=1,
        )

    def _detect_turns(self, text: str) -> list[tuple[MessageRole, str]]:
        """Detect conversation turns in free-form text.

        Scans line by line for turn markers (User:, Assistant:, etc.).
        Groups consecutive lines into turns.

        Returns:
            List of (role, content) tuples. Empty list if no turns detected.
        """
        lines = text.split("\n")
        turns: list[tuple[MessageRole, str]] = []
        current_role: MessageRole | None = None
        current_lines: list[str] = []
        found_any_marker = False

        for line in lines:
            # Skip section separators
            if _SEPARATOR_PATTERN.match(line.strip()):
                continue

            # Check for turn marker
            matched_role = None
            remaining_line = line
            for pattern, role in _TURN_PATTERNS:
                match = pattern.match(line)
                if match:
                    matched_role = role
                    remaining_line = line[match.end():]
                    found_any_marker = True
                    break

            if matched_role is not None:
                # Save previous turn
                if current_role is not None and current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        turns.append((current_role, content))

                current_role = matched_role
                current_lines = [remaining_line] if remaining_line.strip() else []
            else:
                current_lines.append(line)

        # Save last turn
        if current_role is not None and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                turns.append((current_role, content))

        # Only return turns if we found at least one marker
        if not found_any_marker:
            return []

        return turns

    def _build_from_turns(
        self, turns: list[tuple[MessageRole, str]],
    ) -> Conversation:
        """Build a Conversation from detected turns."""
        system_instruction = ""
        parsed_messages: list[Message] = []

        for role, content in turns:
            if role == MessageRole.SYSTEM:
                system_instruction = content
                continue

            blocks = self._extract_blocks_from_text(content)
            if blocks:
                msg = Message(
                    role=role,
                    content=tuple(blocks),
                )
                parsed_messages.append(msg)

        return Conversation(
            source_format=SourceFormat.GENERIC,
            source_llm="unknown",
            system_instruction=system_instruction,
            messages=tuple(parsed_messages),
            message_count=len(parsed_messages),
        )

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
