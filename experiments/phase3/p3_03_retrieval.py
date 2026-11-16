"""Retrieval comparison: dense vs BM25 vs hybrid vs hybrid+rerank. No LLM calls.

Reads results/phase3/questions.jsonl (frozen), writes
results/phase3/retrieval_per_question.jsonl and retrieval_summary.json.
Unanswerable questions have no gold evidence and are excluded from recall.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from src.benchmark.metrics import bootstrap_mean, paired_diff, question_scores
from src.benchmark.retrieval import MODES, RetrievalBench

OUT = Path("results/phase3")
KS = (1, 3, 5, 8)


def main() -> None:
    qs = [json.loads(l) for l in (OUT / "questions.jsonl").read_text().splitlines()]
    qs = [q for q in qs if q["type"] != "unanswerable"]
    bench = RetrievalBench()
    rows = []
    try:
        for q in qs:
            gold = [set(h) for h in q["gold_hops"]]
            rec = {"id": q["id"], "type": q["type"], "split": q["split"]}
            for mode in MODES:
                hits = bench.retrieve(q["question"], mode, top_n=10)
                ids = [h["id"] for h in hits]
                sc = question_scores(ids, gold, KS)
                rec[mode] = {"ids": ids, **sc}
            rows.append(rec)
    finally:
        bench.close()
    (OUT / "retrieval_per_question.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary: dict = {"n_by_split": {}, "metrics": {}, "paired": {}}
    metrics = [f"all_hops@{k}" for k in KS] + [f"hop_recall@{k}" for k in KS] + ["rr"]
    for split in ("test", "dev"):
        for group, sel in (("all", lambda r: True), ("single", lambda r: r["type"] == "single"),
                           ("multi", lambda r: r["type"] != "single")):
            sub = [r for r in rows if r["split"] == split and sel(r)]
            if not sub:
                continue
            summary["n_by_split"][f"{split}/{group}"] = len(sub)
            for mode in MODES:
                for m in metrics:
                    mean, lo, hi = bootstrap_mean(np.array([r[mode][m] for r in sub]))
                    summary["metrics"][f"{split}/{group}/{mode}/{m}"] = [mean, lo, hi]
            for a, b in (("hybrid", "dense"), ("hybrid", "sparse"), ("hybrid_rerank", "hybrid"),
                         ("hybrid_rerank", "dense")):
                for m in ("all_hops@5", "all_hops@8", "rr"):
                    d = paired_diff(np.array([r[a][m] for r in sub]), np.array([r[b][m] for r in sub]))
                    summary["paired"][f"{split}/{group}/{a}-{b}/{m}"] = list(d)
    (OUT / "retrieval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["n_by_split"]))


if __name__ == "__main__":
    main()
