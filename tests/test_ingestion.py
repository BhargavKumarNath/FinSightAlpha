"""
Ingestion coverage: parser + chunker correctness against the real,
committed SEC filing (no synthetic/demo fixtures).

These tests exist to catch the class of drift found in Phase 1: the
committed real filing under data/raw/ must always parse and chunk to
the exact chunk count that's live in Qdrant/BM25/the document registry
(currently 284, per data/.registry_sec_filings.json).
"""
import hashlib
from pathlib import Path

from src.ingestion.chunking import SemanticChunker
from src.ingestion.parsers.registry import ParserRegistry
from src.ingestion.parsers.sec_parser import SECEdgarParser

REAL_FILING = Path(
    "data/raw/sec-edgar-filings/NVDA/10-K/0001045810-26-000021/full-submission.txt"
)

# Must match the chunk_count in data/.registry_sec_filings.json / Qdrant
# points_count / both BM25 pickle corpora (see tests/test_retrieval.py).
LIVE_INDEX_CHUNK_COUNT = 284


def test_real_filing_fixture_exists():
    assert REAL_FILING.exists(), (
        f"Real committed filing missing at {REAL_FILING} — the raw dataset "
        "must remain checked into the repo."
    )


def test_registry_selects_sec_parser_for_real_filing():
    registry = ParserRegistry.default()
    parser = registry.get_parser(str(REAL_FILING))
    assert isinstance(parser, SECEdgarParser)


def test_sec_edgar_parser_extracts_real_metadata():
    parsed = SECEdgarParser().parse(str(REAL_FILING))

    assert parsed.metadata.extra["company_name"] == "NVIDIA CORP"
    assert parsed.metadata.extra["cik"] == "0001045810"
    assert parsed.metadata.extra["accession_number"] == "0001045810-26-000021"
    assert parsed.metadata.extra["submission_type"] == "10-K"
    assert parsed.char_count > 0


def test_sec_edgar_parser_skips_non_text_document_types():
    """SKIP_TYPES (GRAPHIC/ZIP/EXCEL/XML/EX-101.*/JSON) must never make it
    into the extracted text as a '=== DOCUMENT: <TYPE> ===' block."""
    parser = SECEdgarParser()
    parsed = parser.parse(str(REAL_FILING))

    for skip_type in parser.SKIP_TYPES:
        assert f"=== DOCUMENT: {skip_type} ===" not in parsed.content


def test_sec_edgar_parser_content_hash_is_deterministic():
    parser = SECEdgarParser()
    first = parser.parse(str(REAL_FILING))
    second = parser.parse(str(REAL_FILING))
    assert first.content_hash() == second.content_hash()
    assert first.content_hash() == "601f611c367e390b"


def test_semantic_chunker_matches_live_index_chunk_count():
    """Regression guard for the Phase 1 284-vs-288 drift: current code +
    current data/raw/ must always reproduce exactly what's indexed."""
    parsed = SECEdgarParser().parse(str(REAL_FILING))
    chunker = SemanticChunker()  # same defaults as DocumentProcessor/IngestionPipeline

    chunks = chunker.chunk(parsed)

    assert len(chunks) == LIVE_INDEX_CHUNK_COUNT


def test_semantic_chunker_chunk_metadata_is_internally_consistent():
    parsed = SECEdgarParser().parse(str(REAL_FILING))
    chunker = SemanticChunker()
    chunks = chunker.chunk(parsed)

    assert len(chunks) > 0

    indices = [c.metadata.chunk_index for c in chunks]
    assert all(a < b for a, b in zip(indices, indices[1:])), "chunk_index must be strictly increasing"
    assert len(set(indices)) == len(indices), "chunk_index must be unique"

    for chunk in chunks:
        assert chunk.metadata.total_chunks == len(chunks)
        assert len(chunk.text) >= chunker.min_chunk_chars
        expected_hash = hashlib.sha256(chunk.text.encode()).hexdigest()[:12]
        assert chunk.metadata.content_hash == expected_hash
        assert chunk.metadata.document_hash == parsed.content_hash()
