"""Compile every measured number into results/phase3/results.json and SUMMARY.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from src.benchmark.metrics import bootstrap_mean, paired_diff

OUT = Path("results/phase3")


def load(name: str) -> list[dict]:
    p = OUT / name
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def acc(rows: list[dict], key: str = "success") -> list:
    x = np.array([float(r[key]) for r in rows])
    return list(bootstrap_mean(x)) if len(x) else []


def main() -> None:
    qs = {q["id"]: q for q in load("questions.jsonl")}
    res: dict = {"questions": {}, "retrieval": json.loads((OUT / "retrieval_summary.json").read_text())}
    from collections import Counter
    res["questions"]["counts"] = dict(Counter(q["type"] for q in qs.values()))
    res["questions"]["split"] = dict(Counter(q["split"] for q in qs.values()))

    arms = {}
    for mode in ("hybrid_rerank", "dense", "hybrid", "sparse"):
        g = load(f"graded_reader_{mode}.jsonl")
        raw = {r["id"]: r for r in load(f"reader_{mode}.jsonl")}
        if not g:
            arms[mode] = {"status": "not run"}
            continue
        ans = [r for r in g if r["type"] != "unanswerable"]
        un = [r for r in g if r["type"] == "unanswerable"]
        a = {"n": len(g), "n_answerable": len(ans), "n_unanswerable": len(un),
             "answerable_correct": acc(ans), "unanswerable_declined": acc(un, "declined"),
             "false_decline_rate": acc(ans, "declined"),
             "all_hops_in_context": acc(ans, "all_hops"),
             "by_type": {t: acc([r for r in ans if r["type"] == t]) + [len([r for r in ans if r["type"] == t])]
                         for t in ("single", "multi_same", "multi_cross")},
             "mean_total_tokens": float(np.mean([raw[r["id"]]["prompt_tokens"] + raw[r["id"]]["completion_tokens"] for r in g])),
             "median_latency_s": float(np.median([raw[r["id"]]["latency_s"] for r in g]))}
        faith = [r["faith"]["score"] for r in g if r.get("faith") and r["faith"]["score"] is not None]
        if faith:
            a["faithfulness"] = {"n": len(faith), "mean_ci": list(bootstrap_mean(np.array(faith))),
                                 "claims": int(sum(r["faith"]["n_claims"] for r in g if r.get("faith")))}
        arms[mode] = a
    res["reader"] = arms

    # paired comparison on questions both arms completed
    base = {r["id"]: r for r in load("graded_reader_hybrid_rerank.jsonl")}
    dense = {r["id"]: r for r in load("graded_reader_dense.jsonl")}
    common = [i for i in base if i in dense and base[i]["type"] != "unanswerable"]
    if common:
        d = paired_diff(np.array([float(base[i]["success"]) for i in common]),
                        np.array([float(dense[i]["success"]) for i in common]))
        res["reader_paired_rerank_minus_dense_answerable"] = {"n": len(common), "diff_ci": list(d)}
    common_all = [i for i in base if i in dense]
    res["reader_common_questions"] = len(common_all)

    for name in ("window_offline_summary.json", "judge_audit.json"):
        p = OUT / name
        if p.exists():
            res[name.split(".")[0]] = json.loads(p.read_text())
    agent = {}
    for a in ("green_off", "green_on", "red_off", "yellow_off"):
        rows = load(f"agent_{a}.jsonl")
        if not rows:
            agent[a] = {"status": "not run"}
            continue
        g = {r["id"]: r for r in load(f"graded_agent_{a}.jsonl")}
        ans = [g[r["id"]] for r in rows if r["id"] in g and g[r["id"]]["type"] != "unanswerable"]
        un = [g[r["id"]] for r in rows if r["id"] in g and g[r["id"]]["type"] == "unanswerable"]
        fs = [x["faith"]["score"] for x in g.values() if x.get("faith") and x["faith"]["score"] is not None]
        agent[a] = {
            "n": len(rows), "n_errors": sum(1 for r in rows if r.get("error")),
            "answerable_correct": [sum(x["success"] for x in ans), len(ans)],
            "unanswerable_declined": [sum(x["success"] for x in un), len(un)],
            "faithfulness_n": len(fs), "faithfulness_mean": float(np.mean(fs)) if fs else None,
            "mean_input_tokens": float(np.mean([r["input_tokens"] for r in rows])),
            "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rows])),
            "median_latency_s": float(np.median([r["latency_s"] for r in rows])),
            "mean_llm_calls": float(np.mean([r["n_calls"] for r in rows]))}
    res["agent_arms"] = agent
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "retrieval"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
