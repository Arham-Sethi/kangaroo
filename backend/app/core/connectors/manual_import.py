"""File/paste import with auto-detection of conversation format.

The Manual Import connector is the MVP ingestion path. Users can:
    1. Paste raw conversation text
    2. Upload a ChatGPT JSON export
    3. Upload an API response dump
    4. Upload a markdown conversation log

The connector auto-detects the format and runs the full UCS generation
pipeline, returning a complete context ready for shifting.

Usage:
    from app.core.connectors.manual_import import ManualImportConnector

    connector = ManualImportConnector()
    result = connector.import_text("User: Hello\nAssistant: Hi!")
    result = connector.import_json({"messages": [...]})
    result = connector.import_file(file_bytes, filename="chat.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.engine.ucs_generator import GenerationResult, UCSGeneratorPipeline


@dataclass(frozen=True)
class ImportResult:
    """Result from a manual import operation.

    Attributes:
        generation: Full UCS generation result.
        source_type: How the data was imported (text, json, file).
        original_size_bytes: Size of the raw input.
        detected_format: Auto-detected conversation format.
    """

    generation: GenerationResult
    source_type: str
    original_size_bytes: int
    detected_format: str


class ManualImportConnector:
    """Import conversations from paste/upload with auto-detection.

    This is the simplest way to get data into Kangaroo Shift.
    No API keys, no browser extension, no OAuth — just paste or upload.
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

    def import_text(self, text: str) -> ImportResult:
        """Import from raw text (paste).

        Handles:
            - JSON strings (auto-parsed)
            - Markdown conversations (User: ... Assistant: ...)
            - Plain text (treated as single user message)

        Args:
            text: Raw text input.

        Returns:
            ImportResult with generated UCS.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Import text cannot be empty.")

        size = len(text.encode("utf-8"))

        # Try JSON parsing first
        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            try:
                data = json.loads(stripped)
                return self._process(data, "text_json", size)
            except json.JSONDecodeError:
                pass

        # Fall through to generic text parsing
        return self._process(text, "text_plain", size)

    def import_json(self, data: dict | list) -> ImportResult:
        """Import from a parsed JSON object.

        Args:
            data: Parsed JSON data (dict or list).

        Returns:
            ImportResult with generated UCS.

        Raises:
            ValueError: If data is empty.
        """
        if not data:
            raise ValueError("Import data cannot be empty.")

        size = len(json.dumps(data).encode("utf-8"))
        return self._process(data, "json", size)

    def import_file(self, content: bytes, filename: str = "") -> ImportResult:
        """Import from an uploaded file.

        Supports:
            - .json files (parsed as JSON)
            - .txt / .md files (parsed as text)

        Args:
            content: Raw file bytes.
            filename: Original filename for format detection.

        Returns:
            ImportResult with generated UCS.

        Raises:
            ValueError: If content is empty or format unsupported.
        """
        if not content:
            raise ValueError("File content cannot be empty.")

        size = len(content)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Decode bytes to string
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = content.decode("latin-1")

        if ext == "json":
            try:
                data = json.loads(text)
                return self._process(data, "file_json", size)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}") from e

        if ext in ("txt", "md", "markdown", ""):
            return self._process(text, "file_text", size)

        raise ValueError(
            f"Unsupported file format: .{ext}. "
            f"Supported: .json, .txt, .md"
        )

    def _process(
        self,
        data: Any,
        source_type: str,
        size: int,
    ) -> ImportResult:
        """Run the UCS generation pipeline on imported data."""
        result = self._pipeline.generate(data)
        return ImportResult(
            generation=result,
            source_type=source_type,
            original_size_bytes=size,
            detected_format=result.conversation.source_format.value,
        )
