"""Load the five-filing corpus used by the Phase 3 evaluation."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

BM25_PATH = Path("data/p3/bm25_filings5.pkl")
COLLECTION = "filings5"
QDRANT_PATH = "data/qdrant_db_p3"
COMPANY_BY_TICKER = {"NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel", "MSFT": "Microsoft", "AAPL": "Apple"}
_PATH_RE = re.compile(r"sec-edgar-filings[\\/](?P<tkr>[A-Z]+)[\\/]10-K[\\/](?P<acc>[\d-]+)")


@dataclass(frozen=True)
class Chunk:
    idx: int  # position == Qdrant point id == BM25 row
    text: str
    ticker: str
    company: str
    accession: str
    section: str
    content_type: str


def load_chunks() -> list[Chunk]:
    _, meta = pickle.load(open(BM25_PATH, "rb"))
    out: list[Chunk] = []
    for i, m in enumerate(meta):
        md = m["metadata"]
        mt = _PATH_RE.search(md["source_path"])
        assert mt, md["source_path"]
        sp = md.get("section_path") or []
        out.append(Chunk(i, m["text"], mt["tkr"], COMPANY_BY_TICKER[mt["tkr"]], mt["acc"],
                         sp[-1] if sp else "", md.get("content_type", "")))
    return out


def norm(s: str) -> str:
    """Whitespace/case/quote-normalised form used for verbatim-quote checks."""
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()
