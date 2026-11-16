"""Draft the question pool with a model that is not the answering model.

Output: results/phase3/question_drafts.jsonl. Every item carries verbatim quotes;
items whose quotes are not substrings of the stated chunks are rejected here.
Human review happens afterwards in p3_02_freeze_questions.py.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from sentence_transformers import SentenceTransformer

from src.benchmark.corpus import load_chunks, norm
from src.benchmark.llm import chat

DRAFTER = "qwen/qwen3.8-27b"
SEED = 20261109
OUT = Path("results/phase3/question_drafts.jsonl")

SINGLE = (
    "You write evaluation questions about SEC 10-K filings. You are shown ONE passage from the "
    "{company} 10-K. Write ONE question that this passage answers with a specific fact (a number, "
    "name, date, or short phrase). Rules: the question must name {company} and make sense without "
    "the passage; it must not say 'passage' or 'text'; the answer must be short (under 20 words); "
    "give a verbatim quote (<=200 chars, copied exactly from the passage) that contains the answer.\n"
    'Return JSON: {{"question": str, "answer": str, "quote": str}}\n\nPASSAGE:\n{text}'
)
MULTI_SAME = (
    "You write evaluation questions about SEC 10-K filings. You are shown TWO passages (A and B) "
    "from the {company} 10-K. Write ONE question that can only be answered by combining a fact "
    "from A with a fact from B (for example a ratio, a sum, a comparison, or a two-part answer). "
    "The question must name {company}, not mention passages, and the answer must be short "
    "(under 30 words). Give one verbatim quote (<=200 chars) from each passage.\n"
    'Return JSON: {{"question": str, "answer": str, "quote_a": str, "quote_b": str}}\n\n'
    "PASSAGE A:\n{a}\n\nPASSAGE B:\n{b}"
)
MULTI_CROSS = (
    "You write evaluation questions about SEC 10-K filings. You are shown passage A from the "
    "{ca} 10-K and passage B from the {cb} 10-K. Write ONE question comparing the two companies "
    "that needs a fact from A and a fact from B (e.g. which company has the larger X, or how do "
    "they differ on Y). Name both companies, never mention passages, answer under 30 words. Give "
    "one verbatim quote (<=200 chars) from each passage.\n"
    'Return JSON: {{"question": str, "answer": str, "quote_a": str, "quote_b": str}}\n\n'
    "PASSAGE A ({ca}):\n{a}\n\nPASSAGE B ({cb}):\n{b}"
)
UNANS = (
    "You write evaluation questions about SEC 10-K filings that CANNOT be answered from the "
    "filing. You are shown ONE passage from the {company} 10-K. Write ONE plausible question about "
    "a closely related fact that this {company} 10-K does NOT state (for example a figure for a "
    "segment, product, or period the passage does not cover, or an internal number companies "
    "normally do not disclose). It must sound like a real analyst question and name {company}. "
    "Do not ask about the future or opinions.\n"
    'Return JSON: {{"question": str, "why_unanswerable": str}}\n\nPASSAGE:\n{text}'
)


def _ok_quote(q: str, text: str) -> bool:
    return bool(q) and len(q) >= 20 and norm(q) in norm(text)


def main() -> None:
    rng = random.Random(SEED)
    chunks = load_chunks()
    good = [c for c in chunks if 400 <= len(c.text) <= 1800 and re.search(r"\d", c.text)
            and c.content_type in ("narrative", "financial", "table", "")]
    by_co: dict[str, list] = {}
    for c in good:
        by_co.setdefault(c.ticker, []).append(c)
    for v in by_co.values():
        rng.shuffle(v)
    tickers = sorted(by_co)

    def take(n_each: int) -> list:
        return [c for t in tickers for c in by_co[t][:n_each]]

    done = set()
    if OUT.exists():
        done = {json.loads(l)["draft_id"] for l in OUT.read_text().splitlines()}
    fout = OUT.open("a")

    def emit(rec: dict) -> None:
        if rec["draft_id"] in done:
            return
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    # single-hop: 14 per company = 70 drafts for 45 targets
    singles = [c for t in tickers for c in by_co[t][:14]]
    # remaining pool for other types, disjoint
    used = {c.idx for c in singles}
    rest = {t: [c for c in by_co[t] if c.idx not in used] for t in tickers}

    for c in singles:
        did = f"S-{c.idx}"
        if did in done:
            continue
        r = chat(DRAFTER, [{"role": "user", "content": SINGLE.format(company=c.company, text=c.text)}],
                 json_mode=True, tag="draft_single")
        try:
            j = json.loads(r.content)
        except Exception:
            continue
        if _ok_quote(j.get("quote", ""), c.text) and j.get("question") and j.get("answer"):
            emit({"draft_id": did, "type": "single", "company": c.company, "chunk_ids": [c.idx],
                  "question": j["question"], "answer": j["answer"], "quotes": [j["quote"]]})

    # multi-hop same filing: 4 per company = 20 drafts for 15 targets
    for t in tickers:
        pool = rest[t]
        for k in range(4):
            a, b = pool[2 * k], pool[2 * k + 1]
            did = f"MS-{a.idx}-{b.idx}"
            if did in done:
                continue
            r = chat(DRAFTER, [{"role": "user", "content": MULTI_SAME.format(
                company=a.company, a=a.text, b=b.text)}], json_mode=True, tag="draft_multi")
            try:
                j = json.loads(r.content)
            except Exception:
                continue
            if _ok_quote(j.get("quote_a", ""), a.text) and _ok_quote(j.get("quote_b", ""), b.text):
                emit({"draft_id": did, "type": "multi_same", "company": a.company,
                      "chunk_ids": [a.idx, b.idx], "question": j["question"],
                      "answer": j["answer"], "quotes": [j["quote_a"], j["quote_b"]]})

    # cross-company: pair each chunk with the most similar chunk of another company
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    cand = [c for t in tickers for c in rest[t][8:20]]
    vecs = emb.encode([c.text for c in cand], normalize_embeddings=True, show_progress_bar=False)
    pairs = []
    seen = set()
    order = list(range(len(cand)))
    rng.shuffle(order)
    for i in order:
        sims = vecs @ vecs[i]
        for j in np.argsort(-sims):
            if cand[j].ticker != cand[i].ticker and sims[j] > 0.55:
                key = tuple(sorted((cand[i].idx, cand[j].idx)))
                if key not in seen and cand[i].idx not in {p[0].idx for p in pairs} \
                        and cand[j].idx not in {p[1].idx for p in pairs}:
                    seen.add(key)
                    pairs.append((cand[i], cand[j]))
                break
        if len(pairs) >= 16:
            break
    for a, b in pairs:
        did = f"MC-{a.idx}-{b.idx}"
        if did in done:
            continue
        r = chat(DRAFTER, [{"role": "user", "content": MULTI_CROSS.format(
            ca=a.company, cb=b.company, a=a.text, b=b.text)}], json_mode=True, tag="draft_cross")
        try:
            j = json.loads(r.content)
        except Exception:
            continue
        if _ok_quote(j.get("quote_a", ""), a.text) and _ok_quote(j.get("quote_b", ""), b.text):
            emit({"draft_id": did, "type": "multi_cross", "company": f"{a.company}+{b.company}",
                  "chunk_ids": [a.idx, b.idx], "question": j["question"], "answer": j["answer"],
                  "quotes": [j["quote_a"], j["quote_b"]]})

    # unanswerable near-misses: 6 per company = 30 drafts for ~12 targets
    for t in tickers:
        for c in rest[t][20:26]:
            did = f"U-{c.idx}"
            if did in done:
                continue
            r = chat(DRAFTER, [{"role": "user", "content": UNANS.format(company=c.company, text=c.text)}],
                     json_mode=True, tag="draft_unans")
            try:
                j = json.loads(r.content)
            except Exception:
                continue
            if j.get("question"):
                emit({"draft_id": did, "type": "unanswerable", "company": c.company,
                      "chunk_ids": [c.idx], "question": j["question"], "answer": "",
                      "quotes": [], "why": j.get("why_unanswerable", "")})
    fout.close()
    print("done")


if __name__ == "__main__":
    main()
