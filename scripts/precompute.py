# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Build-time precompute step for the Next.js frontend.

Reads the real backend/data sources — the filing registry, the RAGAS
report, and the live optimization config — and writes typed JSON
artifacts to frontend/content/generated/. No metric here is invented:
every artifact carries a `provenance` field, and anything explicitly
mocked upstream (LATENCY_BREAKDOWN, CACHE_TOKEN_SAVINGS in
src/ui/components/data.py) is intentionally left out rather than
passed through. See deployment_roadmap.md §7.

Run with: uv run scripts/precompute.py
"""

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "ui"))

# Dependency-free real sources (no torch/streamlit import chain — see
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


def write_json(name: str, data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def build_corpus() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text())
    filings = []
    total_chunks = 0
    for doc_id, entry in registry.items():
        total_chunks += entry["chunk_count"]
        filings.append(
            {
                "doc_id": doc_id,
                "title": entry["metadata"]["title"],
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
            "note": "Product capability, not a measured metric — only the NVDA 10-K above is actually indexed.",
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
    summary = {
        "num_queries": len(rows),
        "avg_faithfulness": round(float(df["faithfulness"].mean()), 4),
        "avg_answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
    }
    return {
        "provenance": "measured",
        "available": True,
        "rows": rows,
        "summary": summary,
        "caveat": (
            "3-query pilot evaluation. Groq's API does not support n>1, so each "
            "metric reflects a single sampled generation per query rather than "
            "an averaged multi-sample estimate — read these as directional, not "
            "statistically robust."
        ),
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
    print("Done.")


if __name__ == "__main__":
    main()
