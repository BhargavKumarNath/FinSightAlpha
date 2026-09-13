# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "qdrant-client"]
# ///
"""Build-time precompute step for the Next.js frontend.

Reads the real backend/data sources: the filing registry, the RAGAS
report, and the live optimization config, and writes typed JSON
artifacts to frontend/content/generated/. No metric here is invented:
every artifact carries a `provenance` field, and anything explicitly
mocked upstream (LATENCY_BREAKDOWN, CACHE_TOKEN_SAVINGS in
src/ui/components/data.py) is intentionally left out rather than
passed through. See deployment_roadmap.md §7.

Run with: uv run scripts/precompute.py
"""

import ast
import csv
import json
import re
import sys
from pathlib import Path

import polars as pl
from qdrant_client import QdrantClient

_CITATION_RE = re.compile(r"\[Doc\s*(\d+):")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "ui"))

# Dependency-free real sources (no torch/streamlit import chain, see
# src/ui/components/data.py's own module docstring for the same guarantee).
from components.data import (
    BUDGET_TIERS,
    CACHE_CONFIG_ROWS,
    CONTEXT_WINDOW_DETAILS,
    FILING_TYPES,
    MODEL_ROUTER_TABLE,
    PIPELINE_NODES,
    ROUTING_RULES,
)

OUT_DIR = REPO_ROOT / "frontend" / "content" / "generated"
REGISTRY_PATH = REPO_ROOT / "data" / ".registry_sec_filings.json"
RAGAS_PATH = REPO_ROOT / "data" / "reports" / "ragas_evaluation_report.csv"
QDRANT_PATH = REPO_ROOT / "data" / "qdrant_db"
QDRANT_COLLECTION = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _no_em_dash(value):
    """The frontend's writing rules ban em dashes outright. Applied once
    here rather than at every call site, so no upstream string (config
    labels, registry titles, recorded model output) can reintroduce one."""
    if isinstance(value, str):
        return value.replace(" — ", ": ").replace("—", "-")
    if isinstance(value, list):
        return [_no_em_dash(v) for v in value]
    if isinstance(value, dict):
        return {k: _no_em_dash(v) for k, v in value.items()}
    return value


def write_json(name: str, data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(_no_em_dash(data), indent=2, default=str) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def build_corpus() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text())
    filings = []
    total_chunks = 0
    for doc_id, entry in registry.items():
        total_chunks += entry["chunk_count"]
        raw_title = entry["metadata"]["title"]
        company_name = raw_title.split("—")[0].strip()
        filings.append(
            {
                "doc_id": doc_id,
                "title": company_name,
                "source_name": entry["source_name"],
                "content_hash": entry["content_hash"],
                "chunk_count": entry["chunk_count"],
                "indexed_at": entry["indexed_at"],
                "collection": entry["collection"],
            }
        )
    return {
        "filings": {
            "provenance": "measured",
            "filing_count": len(filings),
            "total_chunks": total_chunks,
            "items": filings,
        },
        "supported_filing_types": {
            "provenance": "illustrative",
            "note": "Parser coverage across filing types. Only the NVDA 10-K below is currently indexed.",
            "items": [{"type": f["type"], "desc": f["desc"]} for f in FILING_TYPES],
        },
    }


def build_evaluation() -> dict:
    if not RAGAS_PATH.exists():
        return {
            "provenance": "measured",
            "available": False,
            "rows": [],
            "summary": None,
        }

    df = pl.read_csv(RAGAS_PATH)
    rows = [
        {
            "question": r["question"],
            "faithfulness": round(float(r["faithfulness"]), 4),
            "answer_relevancy": round(float(r["answer_relevancy"]), 4),
        }
        for r in df.iter_rows(named=True)
    ]
    avg_faithfulness = sum(r["faithfulness"] for r in rows) / len(rows)
    avg_answer_relevancy = sum(r["answer_relevancy"] for r in rows) / len(rows)
    summary = {
        "num_queries": len(rows),
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_answer_relevancy": round(avg_answer_relevancy, 4),
    }
    return {
        "provenance": "measured",
        "available": True,
        "rows": rows,
        "summary": summary,
        "caveat": (
            "3-query pilot evaluation. Groq's API does not support n>1, so each "
            "metric reflects a single sampled generation per query rather than "
            "an averaged multi-sample estimate. Read these as directional, not "
            "statistically robust."
        ),
    }


def build_console_replays() -> dict:
    """Real recorded transcripts (question + actual model response + actual
    scores) from the RAGAS report, for the Console page's replay mode. The
    per-stage timings shown around these in the UI are indicative, not
    measured: no historical per-stage log exists for these runs."""
    if not RAGAS_PATH.exists():
        return {"provenance": "measured", "items": []}

    items = []
    with open(RAGAS_PATH, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                contexts = ast.literal_eval(raw.get("retrieved_contexts", "[]"))
            except (ValueError, SyntaxError):
                contexts = []
            response = raw.get("response", "")
            cited = set(_CITATION_RE.findall(response))
            items.append(
                {
                    "question": raw["question"],
                    "response": response,
                    "faithfulness": round(float(raw["faithfulness"]), 4),
                    "answer_relevancy": round(float(raw["answer_relevancy"]), 4),
                    "retrieved_count": len(contexts),
                    "cited_count": len(cited),
                }
            )
    return {"provenance": "measured", "items": items}


def build_vector_index() -> dict:
    """Export the already-embedded corpus straight out of the local Qdrant
    collection, rather than re-embedding with sentence-transformers/torch
    (neither is a frontend dependency, and both are heavy to install just
    for this). The Node chat route bundles this file and does cosine
    similarity + BM25 in memory: the corpus is 284 chunks, well within
    what's sane to ship in a serverless function, and it avoids standing
    up a Qdrant Cloud account (see deployment_roadmap.md §9, Phase 2)."""
    client = QdrantClient(path=str(QDRANT_PATH))
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=10_000,
        with_vectors=True,
        with_payload=True,
    )
    chunks = []
    for p in points:
        payload = p.payload or {}
        meta = payload.get("metadata", {})
        source = meta.get("source", meta.get("source_path", "unknown"))
        raw_vector = p.vector if isinstance(p.vector, list) else []
        vector = [round(float(x), 6) for x in raw_vector if isinstance(x, (int, float))]
        chunks.append(
            {
                "id": p.id,
                "text": payload.get("text", ""),
                "source_name": source.split("/")[-1].split("\\")[-1],
                "chunk_index": meta.get("chunk_index"),
                "content_hash": meta.get("content_hash"),
                "vector": vector,
            }
        )
    chunks.sort(key=lambda c: c["id"])
    return {
        "provenance": "measured",
        "model": EMBEDDING_MODEL,
        "dimensions": len(chunks[0]["vector"]) if chunks else 0,
        "chunks": chunks,
    }


def _strip_color(rows, fields):
    """MODEL_ROUTER_TABLE/CACHE_CONFIG_ROWS/CONTEXT_WINDOW_DETAILS are tuples
    ending in a Streamlit hex color the new design system doesn't reuse."""
    return [dict(zip(fields, row[: len(fields)])) for row in rows]


def build_system() -> dict:
    return {
        "provenance": "measured",
        "pipeline": {
            "nodes": [
                {k: v for k, v in node.items() if k != "color"}
                for node in PIPELINE_NODES
            ],
            "routing_rules": [
                {k: v for k, v in rule.items() if k != "color"}
                for rule in ROUTING_RULES
            ],
        },
        "budget_tiers": [
            {k: v for k, v in tier.items() if k != "color"} for tier in BUDGET_TIERS
        ],
        "model_router_table": _strip_color(
            MODEL_ROUTER_TABLE, ["task", "tier", "model", "reason", "approx_cost"]
        ),
        "model_router_caveat": (
            "Reflects src/optimization/config.py's configured model IDs as of "
            "the last precompute run. Groq has since retired the Llama 3.x "
            "IDs shown here (confirmed against the live /v1/models endpoint, "
            "see deployment_roadmap.md §9 Phase 2) -- the live serverless "
            "chat route now calls openai/gpt-oss-120b and openai/gpt-oss-20b "
            "instead; this table has not been repointed at those."
        ),
        "cache_config": _strip_color(CACHE_CONFIG_ROWS, ["label", "value"]),
        "context_window": _strip_color(
            CONTEXT_WINDOW_DETAILS, ["label", "value", "desc"]
        ),
    }


def main() -> None:
    print("Precomputing static JSON artifacts for the frontend...")
    write_json("corpus.json", build_corpus())
    write_json("evaluation.json", build_evaluation())
    write_json("system.json", build_system())
    write_json("console.json", build_console_replays())
    write_json("vector_index.json", build_vector_index())
    print("Done.")


if __name__ == "__main__":
    main()
