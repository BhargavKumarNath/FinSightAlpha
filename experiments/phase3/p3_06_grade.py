"""Grade result files with the frozen graders.

usage: p3_06_grade.py <records.jsonl> [--faith]
Writes results/phase3/graded_<name>.jsonl (one row per question).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.benchmark.answering import grade_correct, grade_decline, grade_faithfulness
from src.benchmark.corpus import load_chunks, norm
from src.benchmark.llm import DailyLimitError

OUT = Path("results/phase3")


def main(path: str, faith: bool) -> None:
    qs = {q["id"]: q for q in (json.loads(l) for l in (OUT / "questions.jsonl").read_text().splitlines())}
    chunks = load_chunks()
    normed = [norm(c.text) for c in chunks]
    src = Path(path)
    dst = OUT / f"graded_{src.stem}.jsonl"
    done = {json.loads(l)["id"] for l in dst.read_text().splitlines()} if dst.exists() else set()
    with dst.open("a") as f:
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["id"] in done:
                continue
            q = qs[r["id"]]
            ans = r["answer"] or ""
            row = {"id": r["id"], "type": q["type"], "error": r.get("error", "")}
            try:
                if q["type"] == "unanswerable":
                    g = grade_decline(q["question"], ans)
                    row.update(declined=g["declined"], success=g["declined"], judge_reason=g["reason"])
                else:
                    g = grade_correct(q["question"], q["answer"], ans)
                    d = grade_decline(q["question"], ans)
                    row.update(correct=g["correct"], by=g["by"], declined=d["declined"],
                               success=g["correct"], judge_reason=g["reason"])
                ctx = r.get("chunks") or [{"text": chunks[i].text} for i in r.get("chunk_ids", [])]
                if faith and ctx and ans:
                    row["faith"] = grade_faithfulness(ans, ctx)
            except DailyLimitError as e:
                print("DAILY LIMIT", str(e)[:200])
                return
            # context coverage: share of gold hops whose text is present in what the system saw
            if q["gold_hops"]:
                if "chunk_ids" in r:
                    seen = set(r["chunk_ids"])
                    cov = [bool(set(h) & seen) for h in q["gold_hops"]]
                else:
                    txt = " ".join(norm(c["text"]) for c in r.get("chunks", []))
                    cov = [any(normed[i][:200] in txt for i in h) for h in q["gold_hops"]]
                row["hop_coverage"] = sum(cov) / len(cov)
                row["all_hops"] = all(cov)
            for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "latency_s", "loops", "n_calls"):
                if k in r:
                    row[k] = r[k]
            f.write(json.dumps(row) + "\n")
            f.flush()
    print("graded", dst)


if __name__ == "__main__":
    main(sys.argv[1], "--faith" in sys.argv)
