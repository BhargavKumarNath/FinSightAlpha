"""
SEC EDGAR Submission Parser.

Handles full-submission.txt files containing embedded HTML with inline XBRL.
Delegates HTML extraction to HTMLParser for composability.
"""

import re
from pathlib import Path
from typing import List, Optional, Set

from src.ingestion.parsers.base import (
    BaseParser, ContentType, DocumentSection, ParsedDocument, ParsedPage,
)
from src.ingestion.parsers.html_parser import HTMLParser


class SECEdgarParser(BaseParser):
    SKIP_TYPES = {
        "GRAPHIC", "ZIP", "EXCEL", "XML", "EX-101.SCH", "EX-101.CAL",
        "EX-101.DEF", "EX-101.LAB", "EX-101.PRE", "JSON",
    }

    def __init__(self, strip_xbrl: bool = True):
        self._html_parser = HTMLParser(strip_xbrl=strip_xbrl)

    def supported_extensions(self) -> Set[str]:
        return {".txt"}

    def can_parse(self, file_path: str) -> bool:
        return self._is_sec_submission(file_path)

    def parse(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        text_parts, pages = [], []
        page_counter = 1

        header_text = self._extract_header(content)
        if header_text:
            text_parts.append(f"=== SEC FILING HEADER ===\n{header_text}")
            pages.append(ParsedPage(page_number=page_counter, content=header_text, content_type=ContentType.METADATA))
            page_counter += 1

        filing_meta = self._extract_filing_metadata(header_text or "")

        for doc_match in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL):
            doc_block = doc_match.group(1)
            type_match = re.search(r"<TYPE>(.*?)[\n<]", doc_block)
            doc_type = type_match.group(1).strip().upper() if type_match else "UNKNOWN"

            if doc_type in self.SKIP_TYPES or doc_type.startswith("EX-101"):
                continue

            text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc_block, re.DOTALL)
            if not text_match:
                continue

            raw_block = text_match.group(1)
            lower_preview = raw_block[:500].lower()

            if any(tag in lower_preview for tag in ["<html", "<body", "<div", "<table"]):
                extracted = self._html_parser.html_to_text(raw_block)
                extracted = self._clean_text(extracted)
            else:
                extracted = self._clean_text(raw_block)

            if extracted and len(extracted.strip()) > 100:
                text_parts.append(f"\n=== DOCUMENT: {doc_type} ===\n{extracted}")
                pages.append(ParsedPage(page_number=page_counter, content=extracted, content_type=ContentType.FINANCIAL))
                page_counter += 1

        full_text = "\n\n".join(text_parts)
        metadata = self._build_metadata(file_path, title=filing_meta.get("title", "SEC Filing"), extra=filing_meta)
        metadata.compute_hash(full_text)
        sections = self._detect_sections(full_text)

        return ParsedDocument(content=full_text, metadata=metadata, pages=pages, sections=sections)

    @staticmethod
    def _is_sec_submission(file_path: str) -> bool:
        if not file_path.lower().endswith(".txt"):
            return False
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(2000)
            return ("<SEC-DOCUMENT>" in header or "sec-edgar" in file_path.lower()
                    or "<ix:" in header or "ACCESSION NUMBER" in header)
        except Exception:
            return False

    def _extract_header(self, content: str) -> Optional[str]:
        match = re.search(r"<SEC-HEADER>(.*?)</SEC-HEADER>", content, re.DOTALL)
        if not match:
            return None
        lines = [l.strip() for l in match.group(1).strip().split("\n") if l.strip() and not l.strip().startswith("<")]
        return "\n".join(lines) if lines else None

    def _extract_filing_metadata(self, header_text: str) -> dict:
        meta = {}
        patterns = {
            "accession_number": r"ACCESSION NUMBER:\s*(\S+)",
            "submission_type": r"CONFORMED SUBMISSION TYPE:\s*(.+)",
            "filed_date": r"FILED AS OF DATE:\s*(\S+)",
            "period_of_report": r"CONFORMED PERIOD OF REPORT:\s*(\S+)",
            "company_name": r"COMPANY CONFORMED NAME:\s*(.+)",
            "cik": r"CENTRAL INDEX KEY:\s*(\S+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                meta[key] = match.group(1).strip()
        company = meta.get("company_name", "Unknown")
        form = meta.get("submission_type", "Filing")
        meta["title"] = f"{company} — {form}"
        return meta

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        lines = text.split("\n")
        return "\n".join(l for l in lines if len(l.strip()) > 2 and not re.match(r"^[\s\W]+$", l.strip()))
