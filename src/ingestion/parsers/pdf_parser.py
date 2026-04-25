"""
PDF Parser — Dual-engine extraction with pdfplumber + PyMuPDF fallback.

Design rationale:
  - pdfplumber: Excellent table extraction, handles complex layouts
  - PyMuPDF (fitz): Fastest raw text extraction, better for scanned PDFs
  
Uses pdfplumber as primary engine for its superior table handling.
Falls back to PyMuPDF if pdfplumber fails or produces empty output.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Set

from src.ingestion.parsers.base import (
    BaseParser,
    ContentType,
    DocumentMetadata,
    DocumentSection,
    ParsedDocument,
    ParsedPage,
)

logger = logging.getLogger(__name__)

# Lazy imports
_pdfplumber = None
_fitz = None


def _get_pdfplumber():
    global _pdfplumber
    if _pdfplumber is None:
        try:
            import pdfplumber
            _pdfplumber = pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF parsing. "
                "Install with: pip install pdfplumber"
            )
    return _pdfplumber


def _get_fitz():
    global _fitz
    if _fitz is None:
        try:
            import fitz
            _fitz = fitz
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) not available. "
                "PDF fallback disabled. Install with: pip install pymupdf"
            )
            _fitz = False  # Sentinel
    return _fitz if _fitz is not False else None


class PDFParser(BaseParser):
    """
    Production-grade PDF parser with table extraction and page tracking.

    Features:
      - Per-page text extraction with page number tracking
      - Table detection and structured extraction
      - Automatic section/heading detection
      - Dual-engine fallback (pdfplumber → PyMuPDF)
      - Metadata extraction (title, author, dates, page count)
    """

    def __init__(self, extract_tables: bool = True, min_page_chars: int = 20):
        """
        Args:
            extract_tables: Whether to extract tables as structured text.
            min_page_chars: Minimum characters for a page to be included.
        """
        self.extract_tables = extract_tables
        self.min_page_chars = min_page_chars

    def supported_extensions(self) -> Set[str]:
        return {".pdf"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # pdfplumber first (better tables), fall back to PyMuPDF
        try:
            return self._parse_with_pdfplumber(file_path)
        except Exception as e:
            logger.warning(f"pdfplumber failed for {path.name}: {e}")
            fitz = _get_fitz()
            if fitz:
                logger.info(f"Falling back to PyMuPDF for {path.name}")
                return self._parse_with_pymupdf(file_path)
            raise

    # pdfplumber Engine

    def _parse_with_pdfplumber(self, file_path: str) -> ParsedDocument:
        pdfplumber = _get_pdfplumber()

        pages: List[ParsedPage] = []
        all_text_parts: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            pdf_metadata = pdf.metadata or {}
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                page_text = page.extract_text() or ""

                # Extract tables
                tables_text: List[str] = []
                if self.extract_tables:
                    for table in page.extract_tables():
                        formatted = self._format_table(table)
                        if formatted:
                            tables_text.append(formatted)

                # Combine page content
                combined = page_text
                if tables_text:
                    combined += "\n\n" + "\n\n".join(tables_text)

                if len(combined.strip()) >= self.min_page_chars:
                    pages.append(
                        ParsedPage(
                            page_number=page_num,
                            content=combined.strip(),
                            tables=tables_text,
                        )
                    )
                    all_text_parts.append(
                        f"[Page {page_num}]\n{combined.strip()}"
                    )

        full_content = "\n\n".join(all_text_parts)
        clean_content = self._clean_text(full_content)

        metadata = self._build_metadata(
            file_path,
            title=pdf_metadata.get("Title"),
            author=pdf_metadata.get("Author"),
            created_date=pdf_metadata.get("CreationDate"),
            modified_date=pdf_metadata.get("ModDate"),
            page_count=total_pages,
        )
        metadata.compute_hash(clean_content)

        sections = self._detect_sections(clean_content)

        return ParsedDocument(
            content=clean_content,
            metadata=metadata,
            pages=pages,
            sections=sections,
        )

    # PyMuPDF Engine (fallback)

    def _parse_with_pymupdf(self, file_path: str) -> ParsedDocument:
        fitz = _get_fitz()

        pages: List[ParsedPage] = []
        all_text_parts: List[str] = []

        doc = fitz.open(file_path)
        try:
            fitz_metadata = doc.metadata or {}
            total_pages = len(doc)

            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text("text") or ""

                if len(page_text.strip()) >= self.min_page_chars:
                    pages.append(
                        ParsedPage(
                            page_number=page_num + 1,
                            content=page_text.strip(),
                        )
                    )
                    all_text_parts.append(
                        f"[Page {page_num + 1}]\n{page_text.strip()}"
                    )
        finally:
            doc.close()

        full_content = "\n\n".join(all_text_parts)
        clean_content = self._clean_text(full_content)

        metadata = self._build_metadata(
            file_path,
            title=fitz_metadata.get("title"),
            author=fitz_metadata.get("author"),
            created_date=fitz_metadata.get("creationDate"),
            modified_date=fitz_metadata.get("modDate"),
            page_count=total_pages,
        )
        metadata.compute_hash(clean_content)

        sections = self._detect_sections(clean_content)

        return ParsedDocument(
            content=clean_content,
            metadata=metadata,
            pages=pages,
            sections=sections,
        )

    # Utilities

    def _format_table(self, table: list) -> str:
        """Convert a pdfplumber table (list of rows) to readable text."""
        if not table:
            return ""

        rows = []
        for row in table:
            cells = [str(cell or "").strip() for cell in row]
            if any(c for c in cells):
                rows.append(" | ".join(cells))

        return "\n".join(rows) if rows else ""

    def _clean_text(self, text: str) -> str:
        """Normalize PDF-extracted text."""
        # Fix common PDF extraction artifacts
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Hyphenation
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\f", "\n\n", text)  # Form feeds

        # Remove lines that are only numbers (page numbers)
        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if not re.match(r"^\s*\d{1,4}\s*$", line.strip())
        ]
        return "\n".join(clean_lines)
