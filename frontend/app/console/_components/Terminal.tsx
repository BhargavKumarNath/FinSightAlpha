"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { Badge, ProvenanceChip, CitedResponse } from "@/lib/ui";
import { cn } from "@/lib/ui/cn";
import type { ConsoleReplay, ChatStage, ChatStreamEvent, EvidenceStageLog } from "@/lib/data/types";
import { streamChat } from "@/lib/data/chatStream";
import { saveEvidence } from "@/lib/evidence/store";

type Stage = {
  id: ChatStage;
  label: string;
  detail: string;
  color: "indigo" | "emerald" | "amber" | "sky";
};

const STAGES: Stage[] = [
  { id: "plan", label: "PLAN", detail: "Decomposing query into research objectives", color: "indigo" },
  { id: "rewrite", label: "REWRITE", detail: "Generating atomic sub-queries", color: "indigo" },
  { id: "retrieve", label: "RETRIEVE", detail: "Hybrid BM25 + dense vector fusion", color: "emerald" },
  { id: "rerank", label: "RERANK", detail: "Cross-encoder reordering candidates", color: "amber" },
  { id: "reason", label: "REASON", detail: "Citation grounded synthesis", color: "emerald" },
  { id: "reflect", label: "REFLECT", detail: "Hallucination and grounding check", color: "sky" },
  { id: "respond", label: "RESPOND", detail: "Emitting final answer", color: "emerald" },
];

const STAGE_BY_ID = Object.fromEntries(STAGES.map((s) => [s.id, s])) as Record<ChatStage, Stage>;

const COLOR_VARS = {
  indigo: "var(--color-indigo)",
  emerald: "var(--color-emerald)",
  amber: "var(--color-amber)",
  sky: "var(--color-sky)",
} as const;

type LogLine = { stage: Stage; detail: string; ms: number };
type LiveResult = Extract<ChatStreamEvent, { type: "result" }> & { queryId: string };

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function Terminal({ replays }: { replays: ConsoleReplay[] }) {
  const [tab, setTab] = useState<"console" | "trace" | "raw">("console");
  const [query, setQuery] = useState("");
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<LogLine[]>([]);
  const [result, setResult] = useState<ConsoleReplay | LiveResult | "not_found" | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  async function execute(q: string, opts: { scrollOnFinish?: boolean } = {}) {
    const trimmed = q.trim();
    if (!trimmed || running) return;
    setRunning(true);
    setLog([]);
    setResult(null);
    setLiveError(null);
    setTab("console");

    const match = replays.find((r) => r.question.trim().toLowerCase() === trimmed.toLowerCase());

    if (match) {
      for (const stage of STAGES) {
        await sleep(180 + Math.random() * 220);
        const ms = Math.round(90 + Math.random() * 640);
        setLog((prev) => [...prev, { stage, detail: stage.detail, ms }]);
      }
      await sleep(250);
      setResult(match);
      setRunning(false);
      if (opts.scrollOnFinish) bodyRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      return;
    }

    const events: ChatStreamEvent[] = [];

    await streamChat(trimmed, (event) => {
      events.push(event);
      if (event.type === "stage") {
        const meta = STAGE_BY_ID[event.stage];
        if (!meta) return;
        setLog((prev) => [...prev, { stage: meta, detail: event.detail, ms: event.ms }]);
      } else if (event.type === "error") {
        setLiveError(event.message);
      }
    });

    setRunning(false);

    const stageLog: EvidenceStageLog[] = events
      .filter((e): e is Extract<ChatStreamEvent, { type: "stage" }> => e.type === "stage")
      .map((e) => ({ stage: e.stage, detail: e.detail, ms: e.ms }));
    const finalResult = events.find(
      (e): e is Extract<ChatStreamEvent, { type: "result" }> => e.type === "result",
    );

    if (finalResult) {
      const queryId = crypto.randomUUID();
      saveEvidence({
        queryId,
        question: finalResult.question,
        response: finalResult.response,
        plan: finalResult.plan,
        subQueries: finalResult.sub_queries,
        loopCount: finalResult.loop_count,
        reflection: finalResult.reflection,
        retrievedCount: finalResult.retrieved_count,
        citedCount: finalResult.cited_count,
        chunks: finalResult.chunks,
        stages: stageLog,
        createdAt: Date.now(),
      });
      setResult({ ...finalResult, queryId });
    }

    if (opts.scrollOnFinish) bodyRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  useEffect(() => {
    if (!replays[0]) return;
    // Kick off the opening replay once on mount so a real result is on
    // screen before any interaction, rather than an empty terminal. Runs
    // silently, no auto-scroll, so the page doesn't jump on first load.
    const id = requestAnimationFrame(() => {
      execute(replays[0].question);
    });
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- intentionally mount-only
  }, []);

  const isLiveResult = result !== null && result !== "not_found" && !("faithfulness" in result);
  const isReplayResult = result !== null && result !== "not_found" && "faithfulness" in result;

  return (
    <div className="glass-panel overflow-hidden p-0">
      {/* Window chrome */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-rose)]/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-amber)]/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-[var(--color-emerald)]/70" />
          </div>
          <div className="flex gap-1">
            {(["console", "trace", "raw"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  "rounded-md px-2.5 py-1 font-mono text-[11px] uppercase tracking-wide transition-colors",
                  tab === t
                    ? "bg-white/[0.08] text-[var(--color-text)]"
                    : "text-[var(--color-text-dim)] hover:text-[var(--color-text-muted)]",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div
          className="flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider"
          style={{ color: running ? COLOR_VARS.sky : COLOR_VARS.amber }}
        >
          <span
            className="h-1.5 w-1.5 rounded-full pulse-dot"
            style={{
              backgroundColor: running ? COLOR_VARS.sky : COLOR_VARS.amber,
              "--pulse-color": running ? "rgba(56,189,248,0.55)" : "rgba(251,191,36,0.55)",
            } as React.CSSProperties}
          />
          {running ? "Running" : "Standby"}
        </div>
      </div>

      {/* Body */}
      <div ref={bodyRef} className="min-h-[200px] px-5 py-5 font-mono text-[13px]">
        {tab === "console" && (
          <div className="flex flex-col gap-2">
            {log.length === 0 && !running && (
              <p className="text-[var(--color-text-dim)]">
                {"// Select an indexed query below for an instant recorded replay, or type any question to run the live pipeline."}
              </p>
            )}
            <AnimatePresence initial={false}>
              {log.map((line, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="flex items-center gap-3"
                >
                  <span style={{ color: COLOR_VARS[line.stage.color] }}>&gt;</span>
                  <span style={{ color: COLOR_VARS[line.stage.color] }} className="w-20 shrink-0">
                    {line.stage.label}
                  </span>
                  <span className="text-[var(--color-text-muted)]">{line.detail}</span>
                  <span className="ml-auto shrink-0 text-[var(--color-text-dim)]">{line.ms}ms</span>
                </motion.div>
              ))}
            </AnimatePresence>

            {isReplayResult && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 rounded-lg border border-white/10 bg-white/[0.03] p-4"
              >
                <div className="mb-2 flex items-center gap-2">
                  <ProvenanceChip kind="measured" />
                  <span className="text-[11px] text-[var(--color-text-dim)]">recorded execution</span>
                </div>
                <CitedResponse text={(result as ConsoleReplay).response} className="font-sans text-[14px] leading-relaxed" />
                <div className="mt-3 flex flex-wrap gap-4 border-t border-white/[0.06] pt-3 text-[11px] text-[var(--color-text-muted)]">
                  <span>Faithfulness <b className="text-[var(--color-emerald)]">{(result as ConsoleReplay).faithfulness.toFixed(3)}</b></span>
                  <span>Relevancy <b className="text-[var(--color-sky)]">{(result as ConsoleReplay).answer_relevancy.toFixed(3)}</b></span>
                  <span>Retrieved <b className="text-[var(--color-text)]">{(result as ConsoleReplay).retrieved_count}</b></span>
                  <span>Cited <b className="text-[var(--color-text)]">{(result as ConsoleReplay).cited_count}</b></span>
                </div>
              </motion.div>
            )}

            {isLiveResult && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 rounded-lg border border-[var(--color-sky)]/25 bg-white/[0.03] p-4"
              >
                <div className="mb-2 flex items-center gap-2">
                  <ProvenanceChip kind="live" />
                  <span className="text-[11px] text-[var(--color-text-dim)]">
                    live run, {(result as LiveResult).loop_count} reflect loop{(result as LiveResult).loop_count === 1 ? "" : "s"}
                  </span>
                </div>
                <CitedResponse text={(result as LiveResult).response} className="font-sans text-[14px] leading-relaxed" />
                <div className="mt-3 flex flex-wrap items-center gap-4 border-t border-white/[0.06] pt-3 text-[11px] text-[var(--color-text-muted)]">
                  <span>Retrieved <b className="text-[var(--color-text)]">{(result as LiveResult).retrieved_count}</b></span>
                  <span>Cited <b className="text-[var(--color-text)]">{(result as LiveResult).cited_count}</b></span>
                  <Link
                    href={`/evidence/${(result as LiveResult).queryId}`}
                    className="ml-auto font-mono text-[var(--color-indigo)] hover:text-[var(--color-text)]"
                  >
                    View evidence trail &rarr;
                  </Link>
                </div>
              </motion.div>
            )}

            {liveError && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 rounded-lg border border-[var(--color-rose)]/30 bg-[var(--color-rose-dim)] p-4"
              >
                <div className="mb-1.5 flex items-center gap-2">
                  <Badge tone="rose">Pipeline error</Badge>
                </div>
                <p className="font-sans text-sm text-[var(--color-text-muted)]">{liveError}</p>
              </motion.div>
            )}
          </div>
        )}

        {tab === "trace" && (
          <div className="flex flex-col gap-1.5">
            {log.length === 0 ? (
              <p className="text-[var(--color-text-dim)]">{"// Run a query to populate the execution trace."}</p>
            ) : (
              log.map((line, i) => (
                <div key={i} className="flex items-center gap-3 text-[var(--color-text-muted)]">
                  <span className="w-6 text-right text-[var(--color-text-dim)]">{String(i + 1).padStart(2, "0")}</span>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: COLOR_VARS[line.stage.color] }} />
                  <span className="w-20 shrink-0" style={{ color: COLOR_VARS[line.stage.color] }}>{line.stage.label}</span>
                  <span className="flex-1">{line.detail}</span>
                  <span className="text-[var(--color-text-dim)]">{line.ms}ms</span>
                </div>
              ))
            )}
          </div>
        )}

        {tab === "raw" && (
          <pre className="whitespace-pre-wrap text-[var(--color-text-muted)]">
            {result && result !== "not_found" ? JSON.stringify(result, null, 2) : "// Execute a query to inspect its raw response payload."}
          </pre>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-white/[0.06] p-4">
        <div className="flex flex-wrap gap-2 pb-3">
          {replays.map((r) => (
            <button
              key={r.question}
              onClick={() => {
                setQuery(r.question);
                execute(r.question, { scrollOnFinish: true });
              }}
              disabled={running}
              className="rounded-full border border-white/10 bg-white/[0.02] px-3 py-1 text-left text-[11px] text-[var(--color-text-muted)] transition-colors hover:border-[var(--color-indigo)]/40 hover:text-[var(--color-text)] disabled:opacity-40"
            >
              {r.question.length > 56 ? `${r.question.slice(0, 56)}...` : r.question}
            </button>
          ))}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            execute(query, { scrollOnFinish: true });
          }}
          className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 focus-within:border-[var(--color-indigo)]/50"
        >
          <span className="font-mono text-[var(--color-indigo)]">$</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask any question about the indexed filing"
            className="flex-1 bg-transparent font-mono text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-dim)] focus:outline-none"
            disabled={running}
          />
          <button
            type="submit"
            disabled={running || !query.trim()}
            className="rounded-md bg-white/[0.08] px-3 py-1.5 font-mono text-xs text-[var(--color-text)] transition-colors hover:bg-white/[0.14] disabled:opacity-30"
          >
            {running ? "Running" : "Execute"}
          </button>
        </form>
        <Badge tone="neutral" className="mt-3">
          Suggested queries replay a recorded transcript instantly. Anything else runs the live retrieval + reasoning pipeline.
        </Badge>
      </div>
    </div>
  );
}
