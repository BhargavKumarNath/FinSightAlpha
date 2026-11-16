"""Single-pass reader and frozen graders for the Phase 3 evaluation.

READER_PROMPT is copied verbatim from `REASONER_PROMPT` in
src/agents/langgraph_agent.py (importing the agent would load its models), so
the ablation reader and the agent's reasoner follow the same instructions.
The grader prompts are frozen here before any scored run.
"""

from __future__ import annotations

import json
import re

from src.benchmark.llm import LLMResult, chat

JUDGE = "qwen/qwen3.8-27b"
READER = "openai/gpt-oss-20b"

READER_PROMPT = (
    "You are an elite Financial Analyst. Answer the user's query using ONLY the provided retrieved chunks. "
    "CRITICAL RULES:\n"
    "1. You MUST cite the exact [Doc X: source_file] for every claim you make.\n"
    "2. Do not mix sources without citing both.\n"
    "3. If specific data (dollar amounts, percentages, dates) appears in the chunks, ALWAYS include it in your answer.\n"
    "4. If the documents genuinely do not contain the answer, explicitly state so — but first carefully re-read ALL chunks.\n"
    "5. Do not fabricate information.\n"
    "6. Provide specific numbers and details whenever they appear in the context.\n"
    "\nUser Query: {original_query}\n\n"
    "--- Retrieved Context ---\n{context}"
)

CORRECT_PROMPT = (
    "You grade answers to questions about SEC filings. Compare the CANDIDATE answer with the GOLD "
    "answer. The candidate is correct if it states the same facts as the gold answer (numbers equal "
    "after unit conversion, e.g. $26,974 million = $26.97 billion; wording may differ; extra correct "
    "detail is fine). It is incorrect if a key fact is missing, wrong, or the candidate says the "
    "information is unavailable.\n"
    'Return JSON: {{"correct": true|false, "reason": "<one sentence>"}}\n\n'
    "QUESTION: {q}\nGOLD: {gold}\nCANDIDATE: {cand}"
)
DECLINE_PROMPT = (
    "The question below cannot be answered from the documents the system was given. Decide whether "
    "the candidate DECLINED, meaning it says the information is not in the documents or cannot be "
    "determined, and does not assert a specific figure or fact as the answer to the question. A "
    "candidate that gives a confident specific answer, or substitutes a different figure as if it "
    "answered the question, did NOT decline.\n"
    'Return JSON: {{"declined": true|false, "reason": "<one sentence>"}}\n\n'
    "QUESTION: {q}\nCANDIDATE: {cand}"
)
FAITH_PROMPT = (
    "List the factual claims in the ANSWER (numbers, names, dates, causal statements; skip citation "
    "markers and statements that the documents lack information). For each claim say whether the "
    "CONTEXT supports it (the context states it or it follows directly).\n"
    'Return JSON: {{"claims": [{{"claim": "...", "supported": true|false}}]}}\n\n'
    "CONTEXT:\n{ctx}\n\nANSWER:\n{ans}"
)

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _numbers(s: str) -> list[str]:
    return [n.replace(",", "").rstrip(".") for n in _NUM.findall(s)]


def format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[Doc {i}: {c['label']}] (Relevance: {c.get('score', 0):.2f})\n{c['text']}"
                              for i, c in enumerate(chunks, 1))


def read(question: str, chunks: list[dict], model: str = READER, tag: str = "reader") -> LLMResult:
    prompt = READER_PROMPT.format(original_query=question, context=format_context(chunks))
    return chat(model, [{"role": "system", "content": prompt}], max_tokens=900, tag=tag)


def _j(r: LLMResult) -> dict:
    try:
        return json.loads(r.content)
    except Exception:
        m = re.search(r"\{.*\}", r.content, re.S)
        return json.loads(m.group(0)) if m else {}


def grade_correct(question: str, gold: str, cand: str) -> dict:
    """Deterministic number match first (only ever a positive shortcut), else the judge."""
    gn = _numbers(gold)
    if gn and all(n in _numbers(cand) for n in gn):
        return {"correct": True, "by": "numeric_match", "reason": ""}
    r = chat(JUDGE, [{"role": "user", "content": CORRECT_PROMPT.format(q=question, gold=gold, cand=cand)}],
             json_mode=True, max_tokens=300, tag="judge_correct")
    j = _j(r)
    return {"correct": bool(j.get("correct", False)), "by": "judge", "reason": j.get("reason", "")}


def grade_decline(question: str, cand: str) -> dict:
    r = chat(JUDGE, [{"role": "user", "content": DECLINE_PROMPT.format(q=question, cand=cand)}],
             json_mode=True, max_tokens=300, tag="judge_decline")
    j = _j(r)
    return {"declined": bool(j.get("declined", False)), "reason": j.get("reason", "")}


def grade_faithfulness(answer: str, chunks: list[dict]) -> dict:
    ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    r = chat(JUDGE, [{"role": "user", "content": FAITH_PROMPT.format(ctx=ctx, ans=answer)}],
             json_mode=True, max_tokens=900, tag="judge_faith")
    claims = _j(r).get("claims", [])
    n = len(claims)
    sup = sum(1 for c in claims if c.get("supported"))
    return {"n_claims": n, "n_supported": sup, "score": (sup / n) if n else None, "claims": claims}
