import "server-only";
import { retrieve } from "./retrieve";
import { planQuery, rewriteQuery, reason, reflect } from "./groq";
import type { ChatEvent, RetrievedChunk } from "./types";

/**
 * Plan -> Rewrite -> Retrieve -> Reason -> Reflect loop, ported from the
 * LangGraph state machine in src/agents/langgraph_agent.py. Per-session
 * token-budget tiering (GREEN/YELLOW/RED) is skipped: a stateless
 * serverless function has no session to track usage against, so this
 * always runs the GREEN-tier defaults (retrieval top_n=8, max 3 reflect
 * loops) rather than throttling. See deployment_roadmap.md §9, Phase 2.
 */

const RETRIEVAL_TOP_N = 8;
const MAX_REFLECT_LOOPS = 3;
const CITATION_RE = /\[Doc\s*(\d+):/g;

// Cross-encoder (ms-marco-MiniLM) rerank logits, not a 0-1 probability.
// Empirically calibrated against this corpus (single NVIDIA 10-K): genuinely
// relevant NVIDIA chunks score positive (measured: 0.25 to 6.44 across
// several real queries), while chunks retrieved for entities/topics with no
// indexed content (Apple, Tesla, AMD test queries) topped out between -6.9
// and -11.3 -- a wide margin below zero. Chunks below this floor are noise:
// dropping them keeps out-of-corpus sub-queries from diluting context with
// irrelevant text, which both wastes token budget (a real cause of Groq TPM
// rate-limit hits on multi-entity questions) and risks the reasoner weakly
// citing something unrelated.
const MIN_RELEVANCE_SCORE = -1;

const NO_RELEVANT_CONTENT_RESPONSE =
  "No content relevant to this question was found in the indexed filing. This system " +
  "currently covers only NVIDIA's 10-K -- it has no data for other companies, real-time " +
  "market figures (e.g. current P/E ratio), or filings beyond the one indexed document. " +
  "Try a question about what's actually in NVIDIA's 10-K instead.";

type ContextChunk = { docId: number; text: string; source: string; score: number };

function formatContextForReasoner(chunks: ContextChunk[]): string {
  if (chunks.length === 0) return "No relevant documents found.";
  return chunks
    .map((c) => `[Doc ${c.docId}: ${c.source}] (Relevance: ${c.score.toFixed(2)})\n${c.text}`)
    .join("\n\n---\n\n");
}

function formatContextForReflector(chunks: ContextChunk[]): string {
  return chunks.map((c) => `[Doc ${c.docId}]: ${c.text.slice(0, 400)}...`).join("\n\n");
}

function countCitations(text: string): number {
  const ids = new Set<string>();
  for (const match of text.matchAll(CITATION_RE)) ids.add(match[1]);
  return ids.size;
}

export async function* runAgent(query: string): AsyncGenerator<ChatEvent, void, unknown> {
  const contextChunks: ContextChunk[] = [];
  const existingTexts = new Set<string>();
  let docIdCounter = 1;
  let plan = "";
  let reflectionNote = "None";
  let subQueries: string[] = [];
  let draft = "";
  let loopCount = 0;

  try {
    let t0 = Date.now();
    plan = await planQuery(query);
    yield { type: "stage", stage: "plan", detail: plan.split("\n")[0]?.slice(0, 90) || "Research plan formed", ms: Date.now() - t0 };

    let next: "query_rewriter" | "reasoner" | "responder" = "query_rewriter";

    while (next !== "responder") {
      if (next === "query_rewriter") {
        t0 = Date.now();
        subQueries = await rewriteQuery(query, plan, reflectionNote);
        yield {
          type: "stage",
          stage: "rewrite",
          detail: `${subQueries.length} atomic sub-quer${subQueries.length === 1 ? "y" : "ies"} generated`,
          ms: Date.now() - t0,
        };

        t0 = Date.now();
        for (const sq of subQueries) {
          const results = await retrieve(sq, RETRIEVAL_TOP_N);
          for (const r of results) {
            if (r.score <= MIN_RELEVANCE_SCORE) continue;
            if (!existingTexts.has(r.text)) {
              existingTexts.add(r.text);
              contextChunks.push({ docId: docIdCounter, text: r.text, source: r.source, score: r.score });
              docIdCounter += 1;
            }
          }
        }
        contextChunks.sort((a, b) => b.score - a.score);
        yield {
          type: "stage",
          stage: "retrieve",
          detail: `${contextChunks.length} chunks in context, hybrid + cross-encoder rerank`,
          ms: Date.now() - t0,
        };

        if (contextChunks.length === 0) {
          draft = NO_RELEVANT_CONTENT_RESPONSE;
          reflectionNote = "Skipped: no chunk cleared the relevance floor";
          next = "responder";
          continue;
        }
      }

      t0 = Date.now();
      draft = await reason(query, formatContextForReasoner(contextChunks));
      yield { type: "stage", stage: "reason", detail: "Citation-grounded draft synthesized", ms: Date.now() - t0 };

      t0 = Date.now();
      const result = await reflect(formatContextForReflector(contextChunks), draft);
      loopCount += 1;
      const action = result.needsMoreInfo ? "needs_retrieval" : !result.isGrounded ? "needs_rewrite" : "good";
      reflectionNote = `Action=${action}, Reason=${result.feedback}`;
      yield {
        type: "stage",
        stage: "reflect",
        detail: `Grounded: ${result.isGrounded}${action !== "good" ? `, looping (${action})` : ""}`,
        ms: Date.now() - t0,
      };

      if (action === "good" || loopCount >= MAX_REFLECT_LOOPS) {
        next = "responder";
      } else if (action === "needs_retrieval") {
        next = "query_rewriter";
      } else {
        next = "reasoner";
      }
    }

    yield {
      type: "stage",
      stage: "respond",
      detail: contextChunks.length === 0 ? "No chunk cleared the relevance floor; skipped reasoning" : "Finalizing answer",
      ms: 0,
    };
    const publicChunks: RetrievedChunk[] = contextChunks.map((c) => ({
      docId: c.docId,
      text: c.text,
      source: c.source,
      score: c.score,
    }));
    yield {
      type: "result",
      question: query,
      response: draft,
      retrieved_count: contextChunks.length,
      cited_count: countCitations(draft),
      plan,
      sub_queries: subQueries,
      loop_count: loopCount,
      reflection: reflectionNote,
      chunks: publicChunks,
    };
  } catch (err) {
    const raw = err instanceof Error ? err.message : "Unknown pipeline error.";
    console.error("[rag/agent] pipeline error:", raw);
    yield { type: "error", message: friendlyErrorMessage(raw) };
  }
}

/**
 * Groq's free/on-demand tier has an 8000 TPM ceiling per model; a broad
 * question that triggers multiple reflect loops can aggregate enough
 * retrieved context to exceed it (413) or hit a request-rate limit (429).
 * Surface that as guidance rather than raw JSON (deployment_roadmap.md §8:
 * "surface a graceful rate-limited/retry state ... rather than a raw error").
 *
 * Separately, gpt-oss models (groq.ts's MODEL_LIGHT/MODEL_HEAVY) have a
 * built-in browser_search tool baked into their chat template and will
 * sometimes still emit a tool call for it -- typically on queries with a
 * "live/current data" flavor -- even though no tools are configured here.
 * Groq's server then rejects the completion outright (400,
 * "tool_use_failed") since tool_choice is effectively none. `reasoning_effort:
 * "low"` in groq.ts mitigates but doesn't fully prevent this, so it's
 * surfaced here as guidance too, rather than leaking the raw JSON.
 */
function friendlyErrorMessage(raw: string): string {
  if (/rate_limit_exceeded|429|413|request too large|tokens per minute/i.test(raw)) {
    return "The model's rate limit was hit for this question (it needed more context than the current quota allows in one minute). Try a narrower question, or retry in a moment.";
  }
  if (/tool_use_failed|tool choice is none|but model called a tool/i.test(raw)) {
    return "This question seems to need live or external data that isn't part of the indexed filing, which this system can't fetch. Try rephrasing around what's actually in NVIDIA's 10-K.";
  }
  return raw;
}
