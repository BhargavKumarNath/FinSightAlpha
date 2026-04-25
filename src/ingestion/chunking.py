"""
Semantic Chunking Engine with Rich Metadata Enrichment.

Converts ParsedDocuments into DocumentChunks that carry contextual metadata
(section headers, page numbers, content hash, hierarchical paths) through
to the vector store for precision retrieval and filtering.

Design:
  - Section-aware: chunks respect section boundaries when possible
  - Page-aware: PDF page numbers flow through to chunk metadata
  - Overlap: configurable overlap ensures no information is lost at boundaries
  - Quality filtering: removes noise chunks before they enter the index
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

from src.ingestion.parsers.base import (
    ContentType, DocumentSection, ParsedDocument, ParsedPage,
)


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk for retrieval filtering."""
    source_path: str
    source_name: str
    chunk_index: int
    total_chunks: int = 0
    section_header: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    content_type: str = "narrative"
    content_hash: str = ""
    char_count: int = 0
    document_title: Optional[str] = None
    document_hash: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage (backward compatible)."""
        d = asdict(self)
        # Add 'source' alias for backward compatibility with retriever
        d["source"] = self.source_path
        # Remove None values for compact storage
        return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding and indexing."""
    text: str
    metadata: ChunkMetadata

    def to_jsonl_dict(self) -> Dict[str, Any]:
        """Serialize to JSONL-compatible dict (backward compatible)."""
        return {
            "page_content": self.text,
            "metadata": self.metadata.to_dict(),
        }


class SemanticChunker:
    """
    Section-aware chunker that preserves document structure in metadata.

    Processing pipeline:
      1. If document has sections → chunk within section boundaries
      2. If document has pages → track page numbers per chunk
      3. Split long sections/pages at paragraph → sentence boundaries
      4. Apply overlap between consecutive chunks
      5. Filter noise chunks (too short, non-textual)
      6. Enrich each chunk with hierarchical metadata
    """

    def __init__(
        self,
        max_chunk_chars: int = 1500,
        overlap_chars: int = 200,
        min_chunk_chars: int = 80,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, document: ParsedDocument) -> List[DocumentChunk]:
        """
        Convert a ParsedDocument into a list of enriched DocumentChunks.

        Chooses the best chunking strategy based on document structure:
          - Section-based for documents with detected sections
          - Page-based for PDFs without clear sections
          - Flat for simple text documents
        """
        if document.has_sections and len(document.sections) >= 3:
            raw_chunks = self._chunk_by_sections(document)
        elif document.has_pages:
            raw_chunks = self._chunk_by_pages(document)
        else:
            raw_chunks = self._chunk_flat(document)

        # Quality filter
        filtered = [c for c in raw_chunks if not self._is_noise(c.text)]

        # Set total_chunks count
        for chunk in filtered:
            chunk.metadata.total_chunks = len(filtered)

        return filtered

    # Chunking Strategies

    def _chunk_by_sections(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk respecting section boundaries for structured documents."""
        chunks = []
        section_path: List[str] = []

        for section in doc.sections:
            # Maintain hierarchical path
            while section_path and len(section_path) >= section.level:
                section_path.pop()
            section_path.append(section.title)

            if len(section.content) <= self.max_chunk_chars:
                chunks.append(self._make_chunk(
                    text=section.content,
                    doc=doc,
                    chunk_index=len(chunks),
                    section_header=section.title,
                    section_path=list(section_path),
                    page_number=section.page_start,
                    content_type=section.content_type.value,
                ))
            else:
                # Split large sections with overlap
                sub_texts = self._split_with_overlap(section.content)
                for sub_text in sub_texts:
                    chunks.append(self._make_chunk(
                        text=sub_text,
                        doc=doc,
                        chunk_index=len(chunks),
                        section_header=section.title,
                        section_path=list(section_path),
                        page_number=section.page_start,
                        content_type=section.content_type.value,
                    ))

        return chunks

    def _chunk_by_pages(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Chunk with page number tracking for PDF-like documents."""
        chunks = []

        for page in doc.pages:
            if len(page.content) <= self.max_chunk_chars:
                chunks.append(self._make_chunk(
                    text=page.content,
                    doc=doc,
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    content_type=page.content_type.value,
                ))
            else:
                sub_texts = self._split_with_overlap(page.content)
                for sub_text in sub_texts:
                    chunks.append(self._make_chunk(
                        text=sub_text,
                        doc=doc,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                        content_type=page.content_type.value,
                    ))

        return chunks

    def _chunk_flat(self, doc: ParsedDocument) -> List[DocumentChunk]:
        """Simple chunking for unstructured documents."""
        chunks = []
        sub_texts = self._split_with_overlap(doc.content)

        for sub_text in sub_texts:
            chunks.append(self._make_chunk(
                text=sub_text,
                doc=doc,
                chunk_index=len(chunks),
            ))

        return chunks

    # Chunk Construction

    def _make_chunk(
        self,
        text: str,
        doc: ParsedDocument,
        chunk_index: int,
        section_header: Optional[str] = None,
        section_path: Optional[List[str]] = None,
        page_number: Optional[int] = None,
        content_type: str = "narrative",
    ) -> DocumentChunk:
        """Construct a DocumentChunk with enriched metadata."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]

        meta = ChunkMetadata(
            source_path=doc.metadata.source_path,
            source_name=doc.metadata.source_name,
            chunk_index=chunk_index,
            section_header=section_header,
            section_path=section_path or [],
            page_number=page_number,
            content_type=content_type,
            content_hash=content_hash,
            char_count=len(text),
            document_title=doc.metadata.title,
            document_hash=doc.metadata.content_hash,
        )

        return DocumentChunk(text=text.strip(), metadata=meta)

    # Text Splitting

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping chunks at paragraph/sentence boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 > self.max_chunk_chars:
                if current:
                    chunks.append(current)
                    overlap = current[-self.overlap_chars:] if len(current) > self.overlap_chars else ""
                    current = overlap + "\n\n" + para if overlap else para
                else:
                    if len(para) > self.max_chunk_chars:
                        chunks.extend(self._split_at_sentences(para))
                        current = ""
                    else:
                        current = para
            else:
                current = current + "\n\n" + para if current else para

        if current and current.strip():
            chunks.append(current)

        return chunks

    def _split_at_sentences(self, text: str) -> List[str]:
        """Split text at sentence boundaries as last resort."""
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

    # Quality Filters

    def _is_noise(self, text: str) -> bool:
        """Detect chunks that are noise/metadata/artifacts."""
        stripped = text.strip()

        if len(stripped) < self.min_chunk_chars:
            return True

        # HTML remnants
        if len(re.findall(r"<[^>]+>", stripped)) / max(len(stripped.split()), 1) > 0.3:
            return True

        # XBRL schema definitions
        if any(kw in stripped for kw in ["parentTag", "schemaRef", "DefinitionEquity"]):
            return True

        # Pure JSON
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                return True
            except json.JSONDecodeError:
                pass

        # Low alphabetic content ratio
        alpha = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 50 and alpha / len(stripped) < 0.3:
            return True

        return False
