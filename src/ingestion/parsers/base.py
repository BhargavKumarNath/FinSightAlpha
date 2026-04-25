"""
Core Data Types & Abstract Parser Interface.

Defines the canonical document representation that all parsers produce.
Every parser converts its native format into ParsedDocument, enabling
uniform downstream processing (chunking, embedding, indexing).

Architecture note: These types form the "contract" between the parsing
layer and the chunking/indexing layers. Adding a new file format only
requires implementing BaseParser — no changes to downstream code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ContentType(str, Enum):
    """Semantic classification of content within a document."""
    NARRATIVE = "narrative"
    TABLE = "table"
    HEADER = "header"
    FOOTNOTE = "footnote"
    METADATA = "metadata"
    FINANCIAL = "financial"
    LEGAL = "legal"
    UNKNOWN = "unknown"


@dataclass
class ParsedPage:
    """
    A single page from a page-aware document (e.g., PDF).

    For formats without native page concepts (HTML, TXT), the
    entire document is represented as a single page.
    """
    page_number: int
    content: str
    tables: List[str] = field(default_factory=list)
    content_type: ContentType = ContentType.NARRATIVE

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class DocumentSection:
    """
    A semantically meaningful section detected within a document.

    Sections are identified by heading patterns (e.g., "Item 1A",
    "Risk Factors", "## Summary") and preserve hierarchical structure.
    """
    title: str
    level: int  # 1 = top-level, 2 = subsection, etc.
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    content_type: ContentType = ContentType.NARRATIVE

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class DocumentMetadata:
    """
    Rich metadata extracted from the document during parsing.

    Captures provenance, structural, and domain-specific attributes
    that flow through to chunk metadata for retrieval filtering.
    """
    source_path: str
    source_name: str
    file_type: str
    file_size_bytes: int = 0
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    page_count: Optional[int] = None
    section_count: int = 0
    content_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self, content: str) -> str:
        """Compute a deterministic hash for deduplication."""
        self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return self.content_hash


@dataclass
class ParsedDocument:
    """
    Canonical representation of a parsed document.

    This is the universal output of every parser, regardless of input
    format. Contains the full text, page-level content (when available),
    detected sections, and rich metadata.

    Downstream consumers (chunker, indexer) only interact with this type.
    """
    content: str
    metadata: DocumentMetadata
    pages: List[ParsedPage] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def has_pages(self) -> bool:
        return len(self.pages) > 1

    @property
    def has_sections(self) -> bool:
        return len(self.sections) > 0

    def content_hash(self) -> str:
        """Return or compute the document's content hash."""
        if not self.metadata.content_hash:
            self.metadata.compute_hash(self.content)
        return self.metadata.content_hash


class BaseParser(ABC):
    """
    Abstract base class for all document parsers.

    Each parser handles one or more file formats and converts them
    into a canonical ParsedDocument. The parser is responsible for:
      1. Reading the file from disk
      2. Extracting clean text content
      3. Detecting sections/headings
      4. Extracting page-level content (if applicable)
      5. Populating document metadata

    To add a new format, subclass BaseParser and register it with
    ParserRegistry.
    """

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a file into a canonical ParsedDocument.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            ParsedDocument with extracted content, sections, and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is unsupported by this parser.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        """
        Return the set of file extensions this parser handles.

        Extensions should be lowercase with leading dot (e.g., {".pdf", ".PDF"}).
        """
        ...

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions()

    def _build_metadata(self, file_path: str, **kwargs) -> DocumentMetadata:
        """Helper to construct metadata with common fields populated."""
        path = Path(file_path)
        return DocumentMetadata(
            source_path=str(path),
            source_name=path.name,
            file_type=path.suffix.lower(),
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            **kwargs,
        )

    def _detect_sections(
        self, text: str, page_number: Optional[int] = None
    ) -> List[DocumentSection]:
        """
        Detect section boundaries in text using heading patterns.

        Handles common patterns:
          - SEC items: "Item 1A. Risk Factors"
          - Markdown headings: "## Section Title"
          - Numbered sections: "1.2.3 Title"
          - Separator-based: "=== SECTION ==="
        """
        sections = []
        patterns = [
            # SEC 10-K items
            (r"^(Item\s+\d+[A-Za-z]?\.?\s*.+)$", 1),
            # PART headers
            (r"^(PART\s+[IVX]+\.?\s*.*)$", 1),
            # Markdown headings
            (r"^(#{1,3})\s+(.+)$", None),  # level from # count
            # Separator-based (=== TITLE ===)
            (r"^={3,}\s*(.+?)\s*={3,}$", 1),
            # "Table of Contents" style
            (r"^((?:Notes?\s+to|Management.s\s+Discussion|Consolidated\s+).+)$", 2),
        ]

        import re
        lines = text.split("\n")
        current_section_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            for pattern, default_level in patterns:
                match = re.match(pattern, stripped, re.IGNORECASE)
                if match:
                    level = default_level
                    if level is None:
                        # Markdown: count # chars
                        level = len(match.group(1))
                        title = match.group(2).strip()
                    elif len(match.groups()) >= 1:
                        title = match.group(1).strip()
                    else:
                        continue

                    # Only create section if there's meaningful content
                    if i - current_section_start > 2:
                        section_content = "\n".join(
                            lines[current_section_start:i]
                        ).strip()
                        if section_content and len(section_content) > 50:
                            sections.append(
                                DocumentSection(
                                    title=title,
                                    level=level,
                                    content=section_content,
                                    page_start=page_number,
                                )
                            )
                    current_section_start = i
                    break

        # Capture final section
        if current_section_start < len(lines) - 1:
            remaining = "\n".join(lines[current_section_start:]).strip()
            if remaining and len(remaining) > 50:
                title = lines[current_section_start].strip()[:80]
                sections.append(
                    DocumentSection(
                        title=title,
                        level=2,
                        content=remaining,
                        page_start=page_number,
                    )
                )

        return sections
