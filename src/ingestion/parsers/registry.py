"""
Parser Registry — Automatic format detection and parser routing.

Implements the Strategy pattern: registers parsers by extension,
with content-based detection fallback for ambiguous formats (e.g.,
.txt files that are actually SEC submissions).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.ingestion.parsers.base import BaseParser, ParsedDocument

logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Central registry that maps file formats to their parser implementations.

    Resolution order for a file:
      1. Content-based detection (parsers with custom can_parse())
      2. Extension-based lookup
      3. Fallback parser (if configured)

    Usage:
        registry = ParserRegistry.default()
        doc = registry.parse("report.pdf")
    """

    def __init__(self):
        self._parsers: List[BaseParser] = []
        self._ext_map: Dict[str, BaseParser] = {}
        self._fallback: Optional[BaseParser] = None

    def register(self, parser: BaseParser, priority: bool = False) -> "ParserRegistry":
        """
        Register a parser. Priority parsers are checked first for
        content-based detection.

        Returns self for fluent chaining.
        """
        if priority:
            self._parsers.insert(0, parser)
        else:
            self._parsers.append(parser)

        for ext in parser.supported_extensions():
            self._ext_map[ext.lower()] = parser

        return self

    def set_fallback(self, parser: BaseParser) -> "ParserRegistry":
        """Set a fallback parser for unrecognized formats."""
        self._fallback = parser
        return self

    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """
        Find the best parser for a file.

        Checks content-based detection first (for ambiguous extensions),
        then falls back to extension-based lookup.
        """
        # Content-based detection (priority parsers first)
        for parser in self._parsers:
            if hasattr(parser, "can_parse") and parser.can_parse.__func__ is not BaseParser.can_parse:
                # Parser has custom can_parse — use content-based detection
                try:
                    if parser.can_parse(file_path):
                        return parser
                except Exception:
                    continue

        # Extension-based lookup
        ext = Path(file_path).suffix.lower()
        if ext in self._ext_map:
            return self._ext_map[ext]

        return self._fallback

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a file using the best available parser.

        Raises ValueError if no parser can handle the file.
        """
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(
                f"No parser registered for: {file_path} "
                f"(supported: {self.supported_extensions()})"
            )

        logger.info(f"Parsing {Path(file_path).name} with {type(parser).__name__}")
        return parser.parse(file_path)

    def supported_extensions(self) -> Set[str]:
        """Return all supported extensions across registered parsers."""
        return set(self._ext_map.keys())

    def can_handle(self, file_path: str) -> bool:
        """Check if any registered parser can handle this file."""
        return self.get_parser(file_path) is not None

    @classmethod
    def default(cls) -> "ParserRegistry":
        """
        Create a registry pre-loaded with all built-in parsers.

        Parser priority:
          1. SECEdgarParser (content-based, catches .txt SEC filings)
          2. PDFParser
          3. HTMLParser
          4. JSONParser
          5. TextParser (also serves as fallback)
        """
        from src.ingestion.parsers.html_parser import HTMLParser
        from src.ingestion.parsers.pdf_parser import PDFParser
        from src.ingestion.parsers.sec_parser import SECEdgarParser
        from src.ingestion.parsers.text_parser import JSONParser, TextParser

        registry = cls()
        text_parser = TextParser()

        # SEC parser has priority (content-based detection for .txt files)
        registry.register(SECEdgarParser(), priority=True)
        registry.register(PDFParser())
        registry.register(HTMLParser())
        registry.register(JSONParser())
        registry.register(text_parser)
        registry.set_fallback(text_parser)

        return registry
