"""
HTML / XBRL Parser.

Handles HTML, XHTML, and inline XBRL documents common in SEC filings.
Strips markup while preserving semantic content, numeric values from
XBRL tags, and table structures.
"""

import re
from pathlib import Path
from typing import List, Set

from bs4 import BeautifulSoup, NavigableString

from src.ingestion.parsers.base import (
    BaseParser,
    ContentType,
    DocumentSection,
    ParsedDocument,
    ParsedPage,
)


class HTMLParser(BaseParser):
    """
    Parser for HTML / XHTML documents with optional inline XBRL.

    Processing pipeline:
      1. Pre-process XBRL inline tags (extract values, strip containers)
      2. Parse with BeautifulSoup
      3. Remove non-content elements (scripts, styles, hidden)
      4. Convert tables to readable text
      5. Extract clean text with section detection
    """

    def __init__(self, strip_xbrl: bool = True):
        self.strip_xbrl = strip_xbrl

    def supported_extensions(self) -> Set[str]:
        return {".html", ".htm", ".xhtml"}

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        clean_text = self.html_to_text(html_content)
        clean_text = self._clean_text(clean_text)

        metadata = self._build_metadata(
            file_path,
            title=self._extract_title(html_content),
        )
        metadata.compute_hash(clean_text)

        sections = self._detect_sections(clean_text)

        return ParsedDocument(
            content=clean_text,
            metadata=metadata,
            pages=[ParsedPage(page_number=1, content=clean_text)],
            sections=sections,
        )

    # HTML Processing

    def html_to_text(self, html_content: str) -> str:
        """Convert HTML/XHTML with inline XBRL to clean, readable text."""
        if self.strip_xbrl:
            html_content = self._extract_xbrl_values(html_content)

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "meta", "link"]):
            tag.decompose()

        # Remove display:none elements
        for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            el.decompose()

        # Convert tables to readable text
        for table in soup.find_all("table"):
            table_text = self._table_to_text(table)
            table.replace_with(NavigableString(f"\n{table_text}\n"))

        return soup.get_text(separator="\n")

    def _extract_xbrl_values(self, html: str) -> str:
        """Extract values from inline XBRL tags, strip containers."""
        # Replace ix: tags with their text content
        for tag in ["nonFraction", "nonNumeric", "fraction"]:
            html = re.sub(
                rf"<ix:{tag}[^>]*>(.*?)</ix:{tag}>",
                r"\1", html, flags=re.DOTALL,
            )

        # Remove hidden blocks (duplicate visible content)
        html = re.sub(r"<ix:hidden[^>]*>.*?</ix:hidden>", "", html, flags=re.DOTALL)

        # Remove structural XBRL containers
        for tag in ["ix:header", "ix:references", "ix:resources"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL)

        # Strip remaining namespace tags
        html = re.sub(r"</?(?:ix|xbrli|xbrldi|link):[^>]*>", "", html)

        return html

    def _table_to_text(self, table_element) -> str:
        """Convert an HTML table to pipe-delimited text."""
        rows = []
        for tr in table_element.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cell_text = re.sub(r"\s+", " ", td.get_text(strip=True))
                cells.append(cell_text)
            if any(c.strip() for c in cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _extract_title(self, html: str) -> str | None:
        """Extract document title from HTML."""
        soup = BeautifulSoup(html[:5000], "html.parser")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else None

    def _clean_text(self, text: str) -> str:
        """Normalize text: collapse whitespace, remove artifacts."""
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if len(line.strip()) > 2 and not re.match(r"^[\s\W]+$", line.strip())
        ]
        return "\n".join(clean_lines)
