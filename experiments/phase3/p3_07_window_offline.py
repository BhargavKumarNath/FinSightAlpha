"""LLM-free measurement of DynamicContextWindow on the test questions.

For each answerable test question: take the production retrieval's top-8 chunks
(hybrid_rerank, from retrieval_per_question.jsonl), apply the window with its
default settings, and compare (a) estimated context tokens and (b) whether each
gold quote is still present verbatim in the context the reader would receive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from sentence_transformers import SentenceTransformer

from src.benchmark.corpus import load_chunks, norm
from src.benchmark.metrics import bootstrap_mean
from src.optimization.context_window import DynamicContextWindow

OUT = Path("results/phase3")


def main() -> None:
    qs = {q["id"]: q for q in map(json.loads, (OUT / "questions.jsonl").read_text().splitlines())}
    rows = [json.loads(l) for l in (OUT / "retrieval_per_question.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "test"]
    chunks = load_chunks()
    win = DynamicContextWindow(embedding_model=SentenceTransformer("all-MiniLM-L6-v2"))
    res = []
    for r in rows:
        q = qs[r["id"]]
        ids = r["hybrid_rerank"]["ids"][:8]
        cand = [{"text": chunks[i].text, "metadata": {}} for i in ids]
        sel = win.select_chunks(q["question"], cand)
        t_all = " ".join(norm(c["text"]) for c in cand)
        t_sel = " ".join(norm(c["text"]) for c in sel)
        quotes = [norm(x) for x in q["quotes"]]
        res.append({
            "id": r["id"], "type": q["type"],
            "tokens_before": sum(len(c["text"]) for c in cand) // 4,
            "tokens_after": sum(len(c["text"]) for c in sel) // 4,
            "n_before": len(cand), "n_after": len(sel),
            "quotes_before": float(np.mean([x in t_all for x in quotes])),
            "quotes_after": float(np.mean([x in t_sel for x in quotes])),
        })
    (OUT / "window_offline.jsonl").write_text("\n".join(json.dumps(x) for x in res) + "\n")
    tb = np.array([x["tokens_before"] for x in res]); ta = np.array([x["tokens_after"] for x in res])
    qb = np.array([x["quotes_before"] for x in res]); qa = np.array([x["quotes_after"] for x in res])
    summ = {
        "n": len(res),
        "tokens_before_median": float(np.median(tb)), "tokens_after_median": float(np.median(ta)),
        "token_reduction_pct": float(100 * (1 - ta.sum() / tb.sum())),
        "quote_retention_before": bootstrap_mean(qb), "quote_retention_after": bootstrap_mean(qa),
        "quote_retention_drop": bootstrap_mean(qb - qa),
        "chunks_dropped_mean": float(np.mean([x["n_before"] - x["n_after"] for x in res])),
    }
    (OUT / "window_offline_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
