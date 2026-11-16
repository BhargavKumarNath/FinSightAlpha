"""Fixed single-pass reader over four retrieval configurations (test questions only).

Resumable: per-question records are appended to results/phase3/reader_<mode>.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.benchmark.answering import read
from src.benchmark.corpus import load_chunks
from src.benchmark.llm import DailyLimitError
from src.benchmark.retrieval import RetrievalBench
from src.retrieval.labels import source_label

OUT = Path("results/phase3")
ORDER = ["hybrid_rerank", "dense", "hybrid", "sparse"]
TOP_N = 5


def main(modes: list[str]) -> None:
    qs = [q for q in (json.loads(l) for l in (OUT / "questions.jsonl").read_text().splitlines())
          if q["split"] == "test"]
    chunks = load_chunks()
    bench = RetrievalBench()
    try:
        for mode in modes:
            path = OUT / f"reader_{mode}.jsonl"
            done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
            with path.open("a") as f:
                for q in qs:
                    if q["id"] in done:
                        continue
                    hits = bench.retrieve(q["question"], mode, top_n=TOP_N)
                    ctx = [{"text": h["text"], "score": h["score"],
                            "label": source_label({"source_path": f"sec-edgar-filings/{chunks[h['id']].ticker}/10-K/{chunks[h['id']].accession}/x"})}
                           for h in hits]
                    try:
                        r = read(q["question"], ctx, tag=f"reader_{mode}")
                    except DailyLimitError as e:
                        print("DAILY LIMIT", mode, q["id"], str(e)[:200])
                        return
                    f.write(json.dumps({"id": q["id"], "mode": mode, "answer": r.content,
                                        "chunk_ids": [h["id"] for h in hits],
                                        "prompt_tokens": r.prompt_tokens,
                                        "completion_tokens": r.completion_tokens,
                                        "latency_s": r.latency_s}) + "\n")
                    f.flush()
            print("finished", mode)
    finally:
        bench.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
