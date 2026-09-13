"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Card, Badge, ProvenanceChip, SectionHeader, CitedResponse } from "@/lib/ui";
import { getEvidence } from "@/lib/evidence/store";
import type { EvidenceRecord } from "@/lib/data/types";

const STAGE_COLOR: Record<string, string> = {
  plan: "var(--color-indigo)",
  rewrite: "var(--color-indigo)",
  retrieve: "var(--color-emerald)",
  rerank: "var(--color-amber)",
  reason: "var(--color-emerald)",
  reflect: "var(--color-sky)",
  respond: "var(--color-emerald)",
};

const CITATION_RE = /\[Doc\s*(\d+):/g;

function citedDocIds(response: string): Set<number> {
  const ids = new Set<number>();
  for (const match of response.matchAll(CITATION_RE)) ids.add(Number(match[1]));
  return ids;
}

export function EvidenceView({ queryId }: { queryId: string }) {
  const [record, setRecord] = useState<EvidenceRecord | null | "loading">("loading");

  useEffect(() => {
    const id = requestAnimationFrame(() => setRecord(getEvidence(queryId)));
    return () => cancelAnimationFrame(id);
  }, [queryId]);

  if (record === "loading") {
    return <p className="font-mono text-sm text-[var(--color-text-dim)]">Loading evidence trail...</p>;
  }

  if (!record) {
    return (
      <>
        <SectionHeader
          eyebrow="Evidence Trail"
          title="This trail isn't available here."
          description="Evidence trails are stored in this browser only, not on a server: a reload after clearing site data, a different browser, or an older run that's been evicted will all land here. Run a live query in Console to generate a fresh one."
        />
        <Link href="/console" className="text-sm text-[var(--color-indigo)] hover:text-[var(--color-text)]">
          &larr; Back to Console
        </Link>
      </>
    );
  }

  const cited = citedDocIds(record.response);
  const sortedChunks = [...record.chunks].sort((a, b) => b.score - a.score);
  const maxMs = Math.max(...record.stages.map((s) => s.ms), 1);

  return (
    <>
      <SectionHeader
        eyebrow="Evidence Trail"
        title={record.question}
        description="Every stage this specific answer actually went through: the plan, the sub-queries it searched, the chunks it retrieved and cited, and how long each step took. Nothing here is aggregated or averaged, it is this one run."
      />

      <div className="mb-10 flex items-center gap-3">
        <ProvenanceChip kind="live" />
        <span className="font-mono text-xs text-[var(--color-text-dim)]">
          {record.loopCount} reflect loop{record.loopCount === 1 ? "" : "s"} &middot; {record.retrievedCount} retrieved &middot; {record.citedCount} cited
        </span>
      </div>

      <section className="mb-10">
        <h2 className="mb-4 text-lg font-semibold text-[var(--color-text)]">Answer</h2>
        <Card hover={false} glow="indigo">
          <CitedResponse text={record.response} className="text-[15px] leading-relaxed" />
        </Card>
      </section>

      <section className="mb-10">
        <h2 className="mb-4 text-lg font-semibold text-[var(--color-text)]">Reasoning trace</h2>
        <Card hover={false} className="space-y-4">
          <div>
            <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--color-text-dim)]">Plan</span>
            <p className="mt-1 whitespace-pre-line text-sm text-[var(--color-text-muted)]">{record.plan}</p>
          </div>
          <div className="border-t border-white/[0.06] pt-4">
            <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--color-text-dim)]">
              Sub-queries ({record.subQueries.length})
            </span>
            <ul className="mt-2 space-y-1.5">
              {record.subQueries.map((sq, i) => (
                <li key={i} className="flex gap-2 text-sm text-[var(--color-text-muted)]">
                  <span className="text-[var(--color-indigo)]">&rarr;</span>
                  {sq}
                </li>
              ))}
            </ul>
          </div>
          {record.reflection && record.reflection !== "None" && (
            <div className="border-t border-white/[0.06] pt-4">
              <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--color-text-dim)]">
                Final reflection
              </span>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{record.reflection}</p>
            </div>
          )}
        </Card>
      </section>

      <section className="mb-10">
        <h2 className="mb-4 text-lg font-semibold text-[var(--color-text)]">Latency waterfall</h2>
        <Card hover={false} className="space-y-2.5">
          {record.stages.map((s, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="w-20 shrink-0 font-mono text-[11px] uppercase tracking-wide text-[var(--color-text-dim)]">
                {s.stage}
              </span>
              <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.04]">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: STAGE_COLOR[s.stage] ?? "var(--color-indigo)" }}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max((s.ms / maxMs) * 100, 3)}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-[11px] text-[var(--color-text-dim)]">{s.ms}ms</span>
            </div>
          ))}
        </Card>
      </section>

      <section>
        <h2 className="mb-4 text-lg font-semibold text-[var(--color-text)]">
          Retrieved chunks, ranked by reranker score
        </h2>
        <div className="grid gap-3">
          {sortedChunks.map((chunk) => (
            <Card key={chunk.docId} hover={false} className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-[var(--color-indigo)]">Doc {chunk.docId}</span>
                  <span className="font-mono text-[11px] text-[var(--color-text-dim)]">{chunk.source}</span>
                  {cited.has(chunk.docId) && <Badge tone="emerald">Cited</Badge>}
                </div>
                <span className="font-mono text-[11px] text-[var(--color-text-dim)]">score {chunk.score.toFixed(3)}</span>
              </div>
              <p className="text-sm leading-relaxed text-[var(--color-text-muted)]">
                {chunk.text.length > 260 ? `${chunk.text.slice(0, 260)}...` : chunk.text}
              </p>
            </Card>
          ))}
        </div>
      </section>
    </>
  );
}
