"""
Document Processor — Production-Grade Multi-Format Ingestion.

Supports arbitrary file formats (HTML, TXT, PDF, CSV, JSON) with
intelligent parsing, HTML/XBRL stripping, and semantic chunking.

Key capabilities:
  - Auto-detects file format and selects appropriate parser
  - Strips HTML/XBRL markup, preserving meaningful text and numeric values
  - Extracts structured data from XBRL inline elements
  - Chunks by semantic sections with configurable overlap
  - Filters out low-quality chunks (too short, pure metadata, etc.)
"""

import os
import re
import glob
import json
import logging
from typing import Iterator, Dict, Any, List, Optional
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Production-grade document processor with multi-format support.
    
    Designed for plug-and-play ingestion: pass any directory of documents
    and get clean, semantically chunked text ready for embedding.
    """

    # Supported file extensions grouped by parser type
    HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
    TEXT_EXTENSIONS = {".txt", ".md", ".log"}
    JSON_EXTENSIONS = {".json", ".jsonl"}
    ALL_SUPPORTED = HTML_EXTENSIONS | TEXT_EXTENSIONS | JSON_EXTENSIONS

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        overlap_chars: int = 200,
        min_chunk_chars: int = 80,
        strip_xbrl: bool = True,
    ):
        """
        Args:
            max_chunk_chars: Maximum characters per chunk.
            overlap_chars: Overlap between consecutive chunks.
            min_chunk_chars: Minimum characters for a chunk to be kept.
            strip_xbrl: Whether to extract values from inline XBRL tags.
        """
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        self.strip_xbrl = strip_xbrl

    # PUBLIC API

    def process_file(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Process a single file into clean, chunked documents.
        
        Auto-detects file format and applies the appropriate parser.
        Yields dicts with 'page_content' and 'metadata' keys.
        """
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()
        print(f"Processing: {file_path} (format: {ext})")

        # Route to appropriate parser
        if ext in self.HTML_EXTENSIONS:
            raw_text = self._parse_html_file(file_path)
        elif self._is_sec_submission(file_path):
            # SEC EDGAR full-submission.txt files contain embedded HTML
            raw_text = self._parse_sec_submission(file_path)
        elif ext in self.TEXT_EXTENSIONS:
            raw_text = self._parse_text_file(file_path)
        elif ext in self.JSON_EXTENSIONS:
            raw_text = self._parse_json_file(file_path)
        else:
            print(f"  [WARN] Unsupported format {ext}, attempting plain text parse")
            raw_text = self._parse_text_file(file_path)

        # Clean the extracted text
        clean_text = self._clean_text(raw_text)

        if not clean_text or len(clean_text.strip()) < self.min_chunk_chars:
            print(f"  [WARN] No meaningful text extracted from {file_path}")
            return

        # Chunk the clean text
        chunks = self._chunk_text(clean_text)
        source_name = Path(file_path).name

        kept = 0
        for chunk_text in chunks:
            # Quality filter
            if len(chunk_text.strip()) < self.min_chunk_chars:
                continue
            if self._is_noise_chunk(chunk_text):
                continue

            kept += 1
            yield {
                "page_content": chunk_text.strip(),
                "metadata": {
                    "source": file_path,
                    "source_name": source_name,
                    "chunk_chars": len(chunk_text),
                }
            }

        print(f"  Extracted {kept} quality chunks from {source_name}")

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = True,
    ) -> str:
        """
        Processes all supported files in a directory tree and saves
        chunks as JSONL files.

        Returns the output directory path.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect all supported files
        files = self._discover_files(input_dir, recursive=recursive)

        if not files:
            print(f"No supported files found in {input_dir}")
            return output_dir

        print(f"Found {len(files)} files to process")

        for file_path in files:
            rel_path = Path(file_path).relative_to(input_dir)
            safe_name = str(rel_path).replace(os.sep, "_").replace("/", "_")
            output_filename = f"{Path(safe_name).stem}_chunks.jsonl"
            output_filepath = os.path.join(output_dir, output_filename)

            print(f"\nChunking: {rel_path} -> {output_filename}")

            chunk_count = 0
            with open(output_filepath, "w", encoding="utf-8") as out_f:
                for chunk_data in self.process_file(file_path):
                    out_f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
                    chunk_count += 1

            if chunk_count > 0:
                print(f"  Saved {chunk_count} chunks to {output_filename}")
            else:
                # Remove empty JSONL files
                os.remove(output_filepath)
                print(f"  [SKIP] No quality chunks produced")

        return output_dir

    # FILE PARSERS

    def _is_sec_submission(self, file_path: str) -> bool:
        """Detect if a .txt file is actually a SEC EDGAR submission with HTML."""
        if not file_path.lower().endswith(".txt"):
            return False
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.read(2000)
            # SEC submissions start with <SEC-DOCUMENT> or contain XBRL
            return ("<SEC-DOCUMENT>" in header or
                    "sec-edgar" in file_path.lower() or
                    "<ix:" in header or
                    "ACCESSION NUMBER" in header)
        except Exception:
            return False

    def _parse_html_file(self, file_path: str) -> str:
        """Parse an HTML/XHTML file, stripping all markup."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        return self._html_to_text(html_content)

    def _parse_sec_submission(self, file_path: str) -> str:
        """
        Parse SEC EDGAR full-submission.txt files.
        
        These files contain:
          1. SEC headers (plain text metadata)
          2. Embedded HTML/XHTML documents with inline XBRL
          3. JSON/XML attachments
          
        Strategy: Extract each <DOCUMENT> block, parse HTML sections
        with BeautifulSoup, and concatenate clean text.
        """
        print("  Detected SEC EDGAR submission format")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        text_parts = []

        # Extract SEC header metadata
        header_match = re.search(
            r"<SEC-HEADER>(.*?)</SEC-HEADER>",
            content, re.DOTALL
        )
        if header_match:
            header_text = self._clean_sec_header(header_match.group(1))
            if header_text:
                text_parts.append(f"=== SEC FILING HEADER ===\n{header_text}")

        # Extract each embedded document
        doc_pattern = re.compile(
            r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL
        )

        for doc_match in doc_pattern.finditer(content):
            doc_block = doc_match.group(1)

            # Extract document type
            type_match = re.search(r"<TYPE>(.*?)[\n<]", doc_block)
            doc_type = type_match.group(1).strip() if type_match else "UNKNOWN"

            # Skip non-primary documents (exhibits, graphics, schemas, etc.)
            if doc_type.upper() in {
                "GRAPHIC", "ZIP", "EXCEL", "XML", "EX-101.SCH",
                "EX-101.CAL", "EX-101.DEF", "EX-101.LAB", "EX-101.PRE",
            }:
                continue

            # Also skip JSON/XML definition files
            if doc_type.upper().startswith("EX-101") or doc_type.upper() == "JSON":
                continue

            # Extract the text content from <TEXT> blocks
            text_match = re.search(
                r"<TEXT>(.*?)</TEXT>", doc_block, re.DOTALL
            )
            if not text_match:
                continue

            raw_block = text_match.group(1)

            # Determine if block contains HTML
            if "<html" in raw_block.lower() or "<body" in raw_block.lower() or "<div" in raw_block.lower():
                extracted = self._html_to_text(raw_block)
            else:
                extracted = raw_block

            if extracted and len(extracted.strip()) > 100:
                text_parts.append(f"\n=== DOCUMENT: {doc_type} ===\n{extracted}")

        result = "\n\n".join(text_parts)
        print(f"  Extracted {len(result):,} characters of clean text from SEC submission")
        return result

    def _parse_text_file(self, file_path: str) -> str:
        """Parse a plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _parse_json_file(self, file_path: str) -> str:
        """Parse a JSON/JSONL file, extracting text fields."""
        texts = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        # Try JSONL first
        if "\n" in content and content.startswith("{"):
            for line in content.split("\n"):
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", obj.get("page_content", "")))
                    if text:
                        texts.append(str(text))
                except json.JSONDecodeError:
                    continue
        else:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            text = item.get("text", item.get("content", ""))
                            if text:
                                texts.append(str(text))
                elif isinstance(data, dict):
                    texts.append(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                return content

        return "\n\n".join(texts)

    # HTML / XBRL PROCESSING

    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML/XHTML with inline XBRL to clean, readable text.
        
        Strategy:
          1. Process inline XBRL tags to extract numeric values
          2. Convert tables to readable text format
          3. Strip all remaining HTML markup
          4. Normalize whitespace
        """
        # Pre-processing: handle XBRL inline elements before BS4 parsing
        if self.strip_xbrl:
            html_content = self._extract_xbrl_values(html_content)

        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script, style, and hidden elements
        for element in soup.find_all(["script", "style", "meta", "link"]):
            element.decompose()

        # Remove display:none elements
        for element in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            element.decompose()

        # Process tables into readable format
        for table in soup.find_all("table"):
            table_text = self._table_to_text(table)
            table.replace_with(NavigableString(f"\n{table_text}\n"))

        # Extract text
        text = soup.get_text(separator="\n")
        return text

    def _extract_xbrl_values(self, html: str) -> str:
        """
        Extract meaningful values from inline XBRL tags.
        
        Converts: <ix:nonFraction ...>1,508</ix:nonFraction>
        To:       1,508
        
        Also handles: ix:nonNumeric, ix:fraction, etc.
        """
        # Replace ix:nonFraction with just its text content
        html = re.sub(
            r"<ix:nonFraction[^>]*>(.*?)</ix:nonFraction>",
            r"\1",
            html,
            flags=re.DOTALL
        )
        html = re.sub(
            r"<ix:nonNumeric[^>]*>(.*?)</ix:nonNumeric>",
            r"\1",
            html,
            flags=re.DOTALL
        )
        html = re.sub(
            r"<ix:fraction[^>]*>(.*?)</ix:fraction>",
            r"\1",
            html,
            flags=re.DOTALL
        )
        # Remove ix:hidden blocks entirely (they duplicate visible content)
        html = re.sub(
            r"<ix:hidden[^>]*>.*?</ix:hidden>",
            "",
            html,
            flags=re.DOTALL
        )
        # Remove ix:header, ix:references, ix:resources blocks
        for tag in ["ix:header", "ix:references", "ix:resources"]:
            html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>",
                "",
                html,
                flags=re.DOTALL
            )
        # Strip any remaining ix: namespace tags
        html = re.sub(r"</?ix:[^>]*>", "", html)
        # Strip xbrli: tags
        html = re.sub(r"</?xbrli:[^>]*>", "", html)
        html = re.sub(r"</?xbrldi:[^>]*>", "", html)
        html = re.sub(r"</?link:[^>]*>", "", html)

        return html

    def _table_to_text(self, table_element) -> str:
        """Convert an HTML table to readable text with aligned columns."""
        rows = []
        for tr in table_element.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cell_text = td.get_text(strip=True)
                # Clean XBRL artifacts
                cell_text = re.sub(r"\s+", " ", cell_text)
                cells.append(cell_text)
            if any(c.strip() for c in cells):  # Skip empty rows
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    # TEXT CLEANING & CHUNKING

    def _clean_text(self, text: str) -> str:
        """Normalize text: collapse whitespace, remove artifacts."""
        if not text:
            return ""

        # Remove HTML entities that survived parsing
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"&#x[0-9a-fA-F]+;", " ", text)

        # Remove URLs (long HTTP links clutter embeddings)
        text = re.sub(r"https?://\S+", "", text)

        # Collapse multiple newlines to max 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces to single space
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Remove lines that are only whitespace/special chars
        lines = text.split("\n")
        clean_lines = [
            line for line in lines
            if len(line.strip()) > 2 and not re.match(r"^[\s\W]+$", line.strip())
        ]

        return "\n".join(clean_lines)

    def _clean_sec_header(self, header_text: str) -> str:
        """Clean SEC header metadata into readable key-value format."""
        lines = header_text.strip().split("\n")
        clean = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("<"):
                clean.append(stripped)
        return "\n".join(clean)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks at section boundaries.
        
        Strategy: Split at section headers (===, ---, numbered sections)
        first, then split long sections at paragraph boundaries,
        then split long paragraphs at sentence boundaries.
        """
        # First, split by section markers
        sections = re.split(
            r"\n(?=={3,}|\-{3,}|#{1,3}\s|(?:ITEM|PART|Section)\s+\d)",
            text,
            flags=re.IGNORECASE
        )

        chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_chars:
                if section.strip():
                    chunks.append(section)
            else:
                # Split long sections into paragraph-level chunks
                sub_chunks = self._split_with_overlap(section)
                chunks.extend(sub_chunks)

        return chunks

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks at paragraph or sentence boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph would exceed max, finalize current chunk
            if len(current_chunk) + len(para) + 2 > self.max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Overlap: carry forward the tail
                    overlap_text = current_chunk[-self.overlap_chars:] if len(current_chunk) > self.overlap_chars else ""
                    current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                else:
                    # Single paragraph exceeds max — split at sentence boundaries
                    if len(para) > self.max_chunk_chars:
                        sentence_chunks = self._split_at_sentences(para)
                        chunks.extend(sentence_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk)

        return chunks

    def _split_at_sentences(self, text: str) -> List[str]:
        """Split a long paragraph at sentence boundaries."""
        # Split at sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current:
            chunks.append(current)

        return chunks

    # QUALITY FILTERS

    def _is_noise_chunk(self, text: str) -> bool:
        """Detect chunks that are pure noise / metadata / formatting artifacts."""
        # Check for excessive HTML/XML tag remnants
        tag_ratio = len(re.findall(r"<[^>]+>", text)) / max(len(text.split()), 1)
        if tag_ratio > 0.3:
            return True

        # Check for XBRL schema definitions
        if "DefinitionEquity" in text or "parentTag" in text or "schemaRef" in text:
            return True

        # Check for pure numeric dumps (e.g., "1290000000\n2739000000")
        lines = text.strip().split("\n")
        numeric_lines = sum(1 for l in lines if re.match(r"^[\d,.\s\-\$%]+$", l.strip()))
        if len(lines) > 3 and numeric_lines / len(lines) > 0.8:
            return True

        # Check for pure JSON metadata
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                return True  # It's a JSON object, not narrative text
            except json.JSONDecodeError:
                pass

        # Check text has sufficient alphabetic content
        alpha_chars = sum(1 for c in text if c.isalpha())
        if len(text) > 50 and alpha_chars / len(text) < 0.3:
            return True

        return False

    # FILE DISCOVERY

    def _discover_files(self, input_dir: str, recursive: bool = True) -> List[str]:
        """Discover all supported files in a directory tree."""
        files = []
        if recursive:
            for root, _, filenames in os.walk(input_dir):
                for fname in filenames:
                    fpath = os.path.join(root, fname)
                    ext = Path(fname).suffix.lower()
                    if ext in self.ALL_SUPPORTED and os.path.isfile(fpath):
                        files.append(fpath)
        else:
            for fname in os.listdir(input_dir):
                fpath = os.path.join(input_dir, fname)
                ext = Path(fname).suffix.lower()
                if ext in self.ALL_SUPPORTED and os.path.isfile(fpath):
                    files.append(fpath)
        return sorted(files)


if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_directory(input_dir="data/raw", output_dir="data/processed")
