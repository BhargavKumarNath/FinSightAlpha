"""Turn reviewed drafts into the frozen question set.

  --review : print every draft for human reading; for unanswerable drafts also print the
             top-5 hybrid hits so absence of the answer can be checked by eye.
  --freeze : read results/phase3/review.json, resolve gold chunk ids from the verbatim
             quotes, assign dev/test by hash, write results/phase3/questions.jsonl once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.benchmark.corpus import load_chunks, norm

OUT = Path("results/phase3")


def split_of(qid: str) -> str:
    return "dev" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 100 < 22 else "test"


def review() -> None:
    from src.benchmark.retrieval import RetrievalBench
    drafts = [json.loads(l) for l in (OUT / "question_drafts.jsonl").read_text().splitlines()]
    bench = RetrievalBench() if any(d["type"] == "unanswerable" for d in drafts) else None
    for d in drafts:
        print(f"\n### {d['draft_id']} [{d['type']}] {d['company']}\nQ: {d['question']}\nA: {d['answer']}")
        for q in d["quotes"]:
            print(f"   quote: {q}")
        if d["type"] == "unanswerable":
            print(f"   why: {d.get('why', '')}")
            for h in bench.retrieve(d["question"], "hybrid_rerank", top_n=3):
                print(f"   hit[{h['id']}]: {h['text'][:160]!r}")
    if bench:
        bench.close()


def freeze() -> None:
    if (OUT / "questions.jsonl").exists():
        sys.exit("questions.jsonl already exists; the set is frozen. Delete it deliberately to rebuild.")
    rv = json.loads((OUT / "review.json").read_text())
    drafts = {d["draft_id"]: d for d in (json.loads(l) for l in (OUT / "question_drafts.jsonl").read_text().splitlines())}
    chunks = load_chunks()
    normed = [norm(c.text) for c in chunks]
    rows = []
    for did in rv["keep"]:
        d = dict(drafts[did])
        d.update(rv.get("edits", {}).get(did, {}))
        hops = []
        for q in d["quotes"]:
            nq = norm(q)
            ids = [i for i, t in enumerate(normed) if nq in t]
            assert ids, f"{did}: quote not found: {q}"
            hops.append(ids)
        rows.append({"type": d["type"], "company": d["company"], "question": d["question"],
                     "answer": d["answer"], "quotes": d["quotes"], "gold_hops": hops,
                     "origin": did})
    for c in rv.get("compose", []):
        quotes, hops = [], []
        for did, qi in c["from"]:
            q = drafts[did]["quotes"][qi]
            nq = norm(q)
            ids = [i for i, t in enumerate(normed) if nq in t]
            assert ids, f"{did}: quote not found: {q}"
            quotes.append(q)
            hops.append(ids)
        rows.append({"type": c["type"], "company": c["company"], "question": c["question"],
                     "answer": c["answer"], "quotes": quotes, "gold_hops": hops,
                     "origin": "+".join(d for d, _ in c["from"])})
    for k, e in enumerate(rv.get("extra_unanswerable", [])):
        rows.append({"type": "unanswerable", "company": e["company"], "question": e["question"],
                     "answer": "", "quotes": [], "gold_hops": [], "origin": f"X-{k}"})
    for n, r in enumerate(rows, 1):
        r["id"] = f"Q{n:03d}"
        r["split"] = split_of(r["id"])
    (OUT / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from collections import Counter
    print(Counter((r["type"], r["split"]) for r in rows))


if __name__ == "__main__":
    {"--review": review, "--freeze": freeze}[sys.argv[1]]()
