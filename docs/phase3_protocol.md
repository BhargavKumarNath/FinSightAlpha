# Phase 3 evaluation protocol (written before any system was scored)

Date written: 2026-11-09. Status of this file: pre-registration. Deviations are appended at the bottom with the date and the reason, never edited into the sections above.

## Why this exists

The previous evaluation was 3 RAGAS questions over one filing. Its faithfulness average (about 0.60) failed its own 0.70 gate, and the RAGAS package no longer imports in the pinned environment. The backend config also named models (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) that Groq no longer serves; the live model list on 2026-11-09 is `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b` for text. `DynamicContextWindow` was built but never called by the agent. Every chunk carries `source = full-submission.txt`, so a citation cannot tell the five filings apart.

## Corpus

Five latest 10-Ks from SEC EDGAR, downloaded 2026-11-09: NVIDIA (FY2026, accession 0001045810-26-000021), AMD (0000002488-26-000018), Intel (0000050863-26-000011), Microsoft (0001193125-26-323660), Apple (0000320193-26-000026). Indexed with the unchanged `IngestionPipeline` (SemanticChunker 1500/200/80, all-MiniLM-L6-v2, Qdrant cosine, BM25Okapi) into a new collection `filings5`: 1,781 chunks. The original single-filing index is untouched.

## Question set

90 questions, frozen before any system is run on them:
- 45 single-hop: one passage answers it.
- 25 multi-hop: 15 need two passages of one filing; 10 compare two companies.
- 20 unanswerable: the corpus does not contain the answer. Built as near-misses (a metric the filing does not disclose, a fiscal year or company not indexed, a forward-looking figure). Each is checked by keyword search over all 1,781 chunks and by reading the top-10 hybrid hits before it is kept.

Construction: a drafting model (qwen/qwen3.8-27b, not the model that answers) is shown a sampled chunk (or a pair) and writes a question, a short answer and a verbatim evidence quote. A script rejects any item whose quote is not a substring of the stated chunk(s). I then read every item and drop or fix bad ones. Gold evidence for retrieval scoring is the set of chunk ids that contain the verbatim quote. Sampling is stratified by company, seed 20261109, over chunks with at least 200 characters and a digit or a named-entity pattern. The drafter never sees any retrieval output, except in the unanswerable check above.

Dev/test split: `sha256(question_id) mod 100 < 22` is dev. Dev questions may be used to debug the harness and agent. Test questions are scored once per configuration; a configuration is never re-run to improve its number. Infrastructure failures (HTTP errors, timeouts) are retried and counted in a failure column, never dropped.

## Retrieval comparison (no LLM, all questions with gold evidence)

Four configurations on identical candidate pools: dense only, BM25 only, hybrid (RRF k=60), hybrid + cross-encoder rerank (the production path). Candidates per path 50, RRF pool max(2*top_n,20), as in `HybridRetriever.search`. Metrics: recall@{1,3,5,8}, MRR@8, "all gold chunks in top-k" for multi-hop (the stricter metric), computed on test questions only for headline numbers; dev numbers reported separately. 95% CIs by bootstrap over questions (10,000 resamples, seed 0); differences between configurations are paired bootstrap.

## Answer quality (LLM)

- Reader for the retrieval ablation: a fixed single-pass reader (`openai/gpt-oss-20b`, temperature 0, top 5 chunks) so the only thing that changes is the retrieval configuration.
- Full agent (`openai/gpt-oss-120b` heavy, `openai/gpt-oss-20b` light) on the same questions.
- Correctness: answerable questions are scored by a frozen rubric: a deterministic match of the gold short answer (normalised numbers and units, case-folded strings) first; items not decidable that way go to a judge (`qwen/qwen3.8-27b`, temperature 0, sees question, gold answer and candidate answer only). Unanswerable questions are correct only if the system declines; a judge decides decline/answer.
- Faithfulness: judge extracts claims from the answer and checks each against the retrieved chunks the system actually saw; reported as the fraction of supported claims. Applied to the test questions that fit the token budget (see below); the number of questions scored is reported next to the figure.
- The judge prompt is frozen before the first scored run. Judge agreement is audited on 20 items I label by hand; the agreement rate is reported.

Rate limits: the free tier allows 8,000 tokens per minute per model and a daily token cap. All LLM calls go through an on-disk cache and a throttled client, so runs can resume. If the daily cap stops a run, the report states how many questions each arm completed; arms are never compared on different question subsets without saying so.

## Budget tiers

The agent is run with the token budget manager forced into each tier (GREEN, YELLOW, RED) before the question. Reported per tier: total tokens (provider-reported usage, not the 4-chars-per-token estimate), wall-clock latency (median, p90), correctness, loops used. Same question subset in all three tiers.

## DynamicContextWindow decision rule (fixed now)

It is wired in as an agent option and evaluated against the no-window agent on the same questions. Adopt it as the default if the paired difference in correctness has a 95% CI lower bound above -0.05 (non-inferior within 5 points) and median total tokens fall by at least 10%. Otherwise delete the module and its tests, and say why on the page.

## Honesty rules

Numbers are reported as measured, including the ones that make the system look worse. No prompt, threshold or retrieval parameter is changed after seeing test results. Changes needed to fix an infrastructure bug are made on dev questions and logged in "Deviations". The evaluation set, the raw model outputs, the judge decisions and the per-question scores are committed under `results/phase3/`.

## Deviations

1. 2026-11-09, before any system was scored. The frozen question set has 81 questions, not 90: 45 single-hop, 10 two-passage same-filing, 5 cross-company, 21 unanswerable (15 near-misses drafted by the model and kept after a keyword check of the filings, 6 out-of-corpus). Reason: about half of the model-drafted multi-hop questions were contrived (sums of unrelated figures, or questions that referred to "passage A"). Drafts were dropped; 9 same-filing and 3 cross-company questions were instead composed by me from verified verbatim quotes of single-fact drafts (every quote is still checked as a substring of the indexed chunks). 22% of ids land in dev by the stated hash rule (16 dev, 65 test). Unanswerable drafts whose answer might appear in the filing were dropped (e.g. Intel DCAI revenue, NVIDIA Data Center revenue, Intel 14A capex).
2. 2026-11-09, before any system was scored. The LLM client's token reservation per request was lowered (completion reserve max_tokens/2 -> max_tokens/6) because the original over-reserved and halved throughput. This changes pacing only, not outputs.
3. 2026-11-09, after the retrieval comparison (which is LLM-free and already scored once on the test split) and before any LLM-based scoring. The free tier caps tokens per minute (8,000) and per day, so the answer-quality runs are fixed as follows.
   - Reader ablation: the 65 test questions, top-5 chunks from the raw question, reader `openai/gpt-oss-20b`. Arm priority order: hybrid_rerank, dense, hybrid, sparse. Arms run to completion in that order; if the daily cap stops the run, the arms not completed are reported as not run.
   - Full agent: a fixed subset of 24 test questions, taken as the first 24 of the test questions sorted by `sha256("agent:" + id)`. The same 24 are used in every agent arm. Arm priority: GREEN/window-off, GREEN/window-on, RED/window-off, YELLOW/window-off. The window decision rule uses the first two. Tier comparisons use window-off.
   - Faithfulness is graded on the agent arms only.
   - The agent's own multi-query retrieval is scored separately: the share of gold hops present in the context the reasoner received.
   - Retrieval finding recorded before these runs and not acted on: hybrid fusion does not beat BM25 alone on this corpus (paired CI includes 0), and multi-hop all-hop recall@8 is 0.25 for every configuration on a single-shot query. No retrieval parameter is changed after seeing these numbers; any improvement would need a fresh question set.
4. 2026-11-09, after the reader ablation hit the free-tier daily cap and before any agent arm was run. Groq's free tier allows 200,000 tokens per day per model, refilled continuously (about 140 tokens per minute); `openai/gpt-oss-20b` was exhausted by the reader runs (hybrid_rerank arm complete, 65 questions; dense arm stopped at 43 of 65; hybrid and sparse arms not run). Consequences, fixed now:
   - The dense arm is compared with the hybrid_rerank arm only on the 43 questions both completed (the first 43 test questions in id order, not a selection). Hybrid and sparse reader arms are reported as not run.
   - The agent arms keep the shipped routing pair (heavy `openai/gpt-oss-120b`, light `openai/gpt-oss-20b`). A first attempt that swapped the light model for `qwen/qwen3.8-27b` was aborted after 3 test questions: Qwen rejects a request with no user message (the reflector sends only a system message) and has a 1,000 output-tokens-per-minute cap. Those 3 records are kept in `results/phase3/aborted/`; they were infrastructure failures (HTTP 400/429), no answer from them was graded, and the run was not continued. The qwen swap is not used anywhere.
   - The agent subset is the first 12 (not 24) of the pre-registered ordering, because one day of quota (200,000 tokens per model) then covers all four arms. Arm order: GREEN/window-off, GREEN/window-on, RED/window-off, YELLOW/window-off. The 20b bucket refills at about 140 tokens per minute, so these arms run when it has refilled; until then they are reported as not run. Stopping at a daily cap leaves the completed prefix, reported as such.
   - With 12 questions the window decision rule cannot reach its non-inferiority bound statistically; the rule's token criterion is evaluated on the agent runs and, LLM-free, on all 65 test questions; its correctness criterion is reported as inconclusive unless the CI allows a decision. The window stays off by default in that case (no evidence to change the default).
