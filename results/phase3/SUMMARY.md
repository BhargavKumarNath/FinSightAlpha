# Phase 3 evaluation: results (2026-11-09)

Protocol and every deviation: `docs/phase3_protocol.md`. Raw outputs, per-question scores and judge decisions are in this directory. Corpus: five 10-Ks (NVIDIA, AMD, Intel, Microsoft, Apple), 1,781 chunks. Questions: 81 frozen before scoring (45 single-hop, 10 two-passage same filing, 5 two-company, 21 unanswerable); 65 are held-out test, 16 are dev. All headline numbers below are the test split, scored once. Intervals are 95% bootstrap over questions (10,000 resamples, seed 0).

## 1. Retrieval (no LLM; 47 answerable test questions)

Share of questions whose gold passage(s) all appear in the top-k.

| Single-hop (n=35) | @1 | @3 | @5 | @8 |
|---|---|---|---|---|
| dense only | 0.29 | 0.51 | 0.54 | 0.57 [0.40, 0.74] |
| BM25 only | 0.51 | 0.69 | 0.71 | 0.83 [0.69, 0.94] |
| hybrid (RRF) | 0.40 | 0.63 | 0.71 | 0.80 [0.66, 0.91] |
| hybrid + rerank (production) | 0.54 | 0.71 | 0.83 | 0.86 [0.74, 0.97] |

Two-passage questions (n=12): all four configurations reach 0.25 [0.00, 0.50] at k=8 on a single-shot query.

Paired differences over all 47 (all-passages@8): rerank minus dense +0.21 [+0.09, +0.34]; hybrid minus dense +0.17 [+0.04, +0.30]; hybrid minus BM25-only -0.02 [-0.15, +0.11]; rerank minus hybrid +0.04 [-0.06, +0.15]. Mean reciprocal rank, rerank minus hybrid: +0.13 [+0.04, +0.23].

Reading: the cross-encoder is the component that earns its place. Fusing dense and BM25 beats dense alone but does not beat BM25 alone on this corpus (financial filings are heavy on exact figures and names; the dense model is a small general one).

## 2. Answer quality (fixed reader gpt-oss-20b, top-5 chunks)

| | hybrid + rerank (65 test q) |
|---|---|
| answerable questions correct (n=47) | 0.77 [0.64, 0.87] |
| single-hop (n=35) | 0.86 [0.74, 0.97] |
| two passages, same filing (n=8) | 0.38 [0.12, 0.75] |
| two companies (n=4) | 0.75 [0.25, 1.00] |
| unanswerable declined (n=18) | 18 of 18 |
| false decline on answerable | 2 of 47 |
| all gold passages in the reader's context | 0.66 [0.51, 0.79] |
| claims supported by the retrieved context | 61 of 65 claims, 45 answers graded |
| tokens per answer call, mean | 1,866 |

Dense-only retrieval, same reader, on the 37 answerable questions both arms completed (the first 43 test questions in id order; 35 are single-hop): correct 0.59 [0.43, 0.76] vs 0.81 for hybrid + rerank on the same 37; paired difference +0.22 [+0.05, +0.38]. Dense retrieval makes the reader decline on 30% of answerable questions (11 of 37).

The ceiling is retrieval. Of the 11 answerable questions the reader got wrong, 10 had at least one gold passage missing from its context; the 11th had everything and misspelled "Deloitte" as "Eloitte".

## 3. DynamicContextWindow (LLM-free, 47 answerable test questions, top-8 chunks)

Default settings change median context from 2,523 to 2,486 estimated tokens (-2.1%), drop 0 chunks, and retain exactly the same gold quotes (0.78 before and after). The pre-registered rule needs a token reduction of at least 10%, so on single-query contexts the module is a no-op. The agent-level measurement (multi-sub-query contexts) is pending, see below.

## 4. Judge audit

20 judge decisions (judge qwen3.8-27b) sampled with seed 7 were read against the question, gold answer and candidate by the author's AI assistant: 20 of 20 agree; one is borderline (Q030, a one-letter misspelling of "Deloitte" marked incorrect). This is not an independent human audit; a human spot check is recommended.

## 5. Agent tier runs (2026-11-10) and what is still not measured

12 held-out questions per arm (7 answerable, 5 unanswerable), tier forced, provider-reported tokens. No intervals: n is too small.

| Arm | Correct (of 7) | Declined (of 5) | Input tok/question | Output tok | Median latency | Calls | Failed runs |
|---|---|---|---|---|---|---|---|
| GREEN, window off | 5 | 4 | 9,580 | 981 | 34 s | 5.7 | 4 (rate limit) |
| GREEN, window on | 6 | 4 | 11,294 | 956 | 40 s | 5.8 | 1 (planner tool call) |
| YELLOW, window off (10 of 12 questions) | 5 of 5 | 5 | 4,415 | 422 | 17 s | 3.0 | 0 |
| RED, window off | 6 | 5 | 3,737 | 398 | 30 s | 3.0 | 0 |

- 4 GREEN/off runs hit Groq's 8,000 tokens/min cap on gpt-oss-120b and returned a fallback answer; they are kept as measured, not re-run. YELLOW stopped at 10 of 12 when gpt-oss-120b reached its daily limit.
- DynamicContextWindow rule (at least 10% token saving): window on raised input tokens by 18%. Not met; flag stays off; the module is a deletion candidate. The off arm lost runs to rate limits, so the size of the difference is uncertain.
- Agent faithfulness is incomplete: the judge (qwen3.8-27b) rejects the largest agent contexts (input cap 7,000 tokens/min), so the green/off arm has only 2 graded answers. No judge or truncation change was made.
- Still not run: reader arms for hybrid-only and BM25-only; dense arm covers 43 of 65; faithfulness for reader covers 45 of 65. YELLOW questions 11 and 12 need gpt-oss-120b quota.

## 6. Reproduce

```
uv run python experiments/phase3/p3_03_retrieval.py            # retrieval, no LLM
uv run python experiments/phase3/p3_04_reader_ablation.py      # needs GROQ_API_KEY, resumable
uv run python experiments/phase3/p3_06_grade.py results/phase3/reader_hybrid_rerank.jsonl --faith
uv run python experiments/phase3/p3_07_window_offline.py
uv run python experiments/phase3/p3_08_report.py
```
