"""
Parser Framework — Pluggable, format-aware document parsing.

Provides a unified interface for parsing diverse document formats
into a canonical representation with rich metadata.
"""

from src.ingestion.parsers.base import (
    BaseParser,
    ParsedDocument,
    ParsedPage,
    DocumentSection,
    DocumentMetadata,
)
from src.ingestion.parsers.registry import ParserRegistry
from src.ingestion.parsers.html_parser import HTMLParser
from src.ingestion.parsers.pdf_parser import PDFParser
from src.ingestion.parsers.text_parser import TextParser, JSONParser
from src.ingestion.parsers.sec_parser import SECEdgarParser

__all__ = [
    "BaseParser",
    "ParsedDocument",
    "ParsedPage",
    "DocumentSection",
    "DocumentMetadata",
    "ParserRegistry",
    "HTMLParser",
    "PDFParser",
    "TextParser",
    "JSONParser",
    "SECEdgarParser",
]
