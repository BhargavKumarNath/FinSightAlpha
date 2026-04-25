"""
Text & JSON Parsers.

Simple parsers for plain text, markdown, and structured JSON/JSONL files.
"""

import json
import re
from pathlib import Path
from typing import Set

from src.ingestion.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
)


class TextParser(BaseParser):
    """Parser for plain text and markdown files."""

    def supported_extensions(self) -> Set[str]:
        return {".txt", ".md", ".log", ".csv", ".tsv"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        clean = self._clean_text(content)
        metadata = self._build_metadata(file_path)
        metadata.compute_hash(clean)

        sections = self._detect_sections(clean)

        return ParsedDocument(
            content=clean,
            metadata=metadata,
            pages=[ParsedPage(page_number=1, content=clean)],
            sections=sections,
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


class JSONParser(BaseParser):
    """
    Parser for JSON and JSONL files.

    Extracts text from common fields: 'text', 'content', 'page_content',
    'body', 'description'. For nested structures, flattens to readable text.
    """

    TEXT_FIELDS = {"text", "content", "page_content", "body", "description", "abstract"}

    def supported_extensions(self) -> Set[str]:
        return {".json", ".jsonl"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read().strip()

        texts = []

        # Try JSONL (one JSON object per line)
        if "\n" in raw and raw.startswith("{"):
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    extracted = self._extract_text(obj)
                    if extracted:
                        texts.append(extracted)
                except json.JSONDecodeError:
                    continue
        else:
            # Single JSON document
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for item in data:
                        extracted = self._extract_text(item)
                        if extracted:
                            texts.append(extracted)
                elif isinstance(data, dict):
                    extracted = self._extract_text(data)
                    texts.append(extracted or json.dumps(data, indent=2))
            except json.JSONDecodeError:
                texts.append(raw)

        content = "\n\n".join(texts)
        metadata = self._build_metadata(file_path)
        metadata.compute_hash(content)

        return ParsedDocument(
            content=content,
            metadata=metadata,
            pages=[ParsedPage(page_number=1, content=content)],
        )

    def _extract_text(self, obj) -> str | None:
        """Extract text from a JSON object by checking common field names."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for field in self.TEXT_FIELDS:
                if field in obj and isinstance(obj[field], str):
                    return obj[field]
            # Fallback: concatenate all string values
            parts = [
                f"{k}: {v}" for k, v in obj.items()
                if isinstance(v, (str, int, float)) and len(str(v)) > 5
            ]
            return "\n".join(parts) if parts else None
        return None
