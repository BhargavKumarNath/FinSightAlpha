import Link from "next/link";
import { getSystemData, getEvaluationData, getCorpusData } from "@/lib/data/loaders";
import { SectionHeader, Card, Badge, ProvenanceChip, Stat } from "@/lib/ui";
import { PipelineGraph } from "./_components/PipelineGraph";

const TIER_TONE = { GREEN: "emerald", YELLOW: "amber", RED: "rose" } as const;

export default function SystemPage() {
  const data = getSystemData();
  const evaluation = getEvaluationData();
  const corpus = getCorpusData();
  const {
    pipeline,
    budget_tiers,
    model_router_table,
    model_router_caveat,
    cache_config,
    context_window,
  } = data;

  return (
    <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6 sm:py-20">
      <SectionHeader
        eyebrow="System"
        title="The mechanics behind every answer."
        description="This is the architecture reference: seven governed stages, budget rules, and routing logic. For what it actually produces, see Console and Evaluation."
      />

      <section className="mb-16">
        <div className="mb-5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-wider text-[var(--color-text-dim)]">
              What this architecture has produced
            </span>
            <ProvenanceChip kind="measured" />
          </div>
          <Link href="/evaluation" className="text-xs text-[var(--color-indigo)] hover:text-[var(--color-text)]">
            Full evaluation &rarr;
          </Link>
        </div>
        <div className="grid gap-4 sm:grid-cols-4">
          <Card>
            <Stat label="Faithfulness, avg" value={evaluation.summary?.avg_faithfulness ?? 0} decimals={3} accent="emerald" />
          </Card>
          <Card>
            <Stat label="Relevancy, avg" value={evaluation.summary?.avg_answer_relevancy ?? 0} decimals={3} accent="indigo" />
          </Card>
          <Card>
            <Stat label="Filings indexed" value={corpus.filings.filing_count} />
          </Card>
          <Card>
            <Stat label="Chunks indexed" value={corpus.filings.total_chunks} />
          </Card>
        </div>
      </section>

      <section className="mb-16">
        <div className="mb-5 flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--color-text)]">Execution graph</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <PipelineGraph nodes={pipeline.nodes} routingRules={pipeline.routing_rules} />
      </section>

      <section className="mb-16">
        <div className="mb-5 flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--color-text)]">Budget governance</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          {budget_tiers.map((tier) => (
            <Card key={tier.tier} glow={TIER_TONE[tier.tier as keyof typeof TIER_TONE] ?? "indigo"}>
              <div className="flex items-center justify-between">
                <Badge tone={TIER_TONE[tier.tier as keyof typeof TIER_TONE] ?? "neutral"}>
                  {tier.tier}
                </Badge>
                <span className="font-mono text-[11px] text-[var(--color-text-dim)]">
                  {tier.range}
                </span>
              </div>
              <p className="mt-3 text-sm text-[var(--color-text-muted)]">{tier.desc}</p>
              <dl className="mt-4 space-y-1.5 border-t border-white/[0.06] pt-4 text-xs">
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">Max loops</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.loops}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">Retrieval depth</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.top_n}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">Model</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.model}</dd>
                </div>
              </dl>
              {tier.skips.length > 0 && (
                <p className="mt-3 font-mono text-[11px] text-[var(--color-rose)]">
                  Bypasses: {tier.skips.join(", ")}
                </p>
              )}
            </Card>
          ))}
        </div>
      </section>

      <section className="mb-16">
        <div className="mb-5 flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--color-text)]">Model routing table</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <Card hover={false} className="overflow-x-auto p-0">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-white/[0.06] text-xs uppercase tracking-wider text-[var(--color-text-dim)]">
                <th className="px-5 py-3 font-normal">Task</th>
                <th className="px-5 py-3 font-normal">Applies at</th>
                <th className="px-5 py-3 font-normal">Model</th>
                <th className="px-5 py-3 font-normal">Rationale</th>
                <th className="px-5 py-3 font-normal">Est. cost</th>
              </tr>
            </thead>
            <tbody>
              {model_router_table.map((row, i) => (
                <tr key={i} className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02]">
                  <td className="px-5 py-3 text-[var(--color-text)]">{row.task}</td>
                  <td className="px-5 py-3 font-mono text-xs text-[var(--color-text-muted)]">{row.tier}</td>
                  <td className="px-5 py-3 font-mono text-xs text-[var(--color-indigo)]">{row.model}</td>
                  <td className="px-5 py-3 text-[var(--color-text-muted)]">{row.reason}</td>
                  <td className="px-5 py-3 font-mono text-xs text-[var(--color-text-dim)]">{row.approx_cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
        {model_router_caveat && (
          <p className="mt-3 text-xs leading-relaxed text-[var(--color-text-dim)]">
            <Badge tone="amber" className="mr-1.5">
              Stale
            </Badge>
            {model_router_caveat}
          </p>
        )}
      </section>

      <section className="grid gap-6 sm:grid-cols-2">
        <div>
          <div className="mb-5 flex items-center gap-2">
            <h2 className="text-lg font-semibold text-[var(--color-text)]">Response cache</h2>
            <ProvenanceChip kind="measured" />
          </div>
          <Card>
            <dl className="space-y-3 text-sm">
              {cache_config.map((row) => (
                <div key={row.label} className="flex justify-between gap-4 border-b border-white/[0.04] pb-3 last:border-0 last:pb-0">
                  <dt className="text-[var(--color-text-muted)]">{row.label}</dt>
                  <dd className="text-right font-mono text-[var(--color-text)]">{row.value}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
        <div>
          <div className="mb-5 flex items-center gap-2">
            <h2 className="text-lg font-semibold text-[var(--color-text)]">Context window</h2>
            <ProvenanceChip kind="measured" />
          </div>
          <Card>
            <dl className="space-y-3 text-sm">
              {context_window.map((row) => (
                <div key={row.label} className="flex justify-between gap-4 border-b border-white/[0.04] pb-3 last:border-0 last:pb-0">
                  <dt>
                    <span className="text-[var(--color-text-muted)]">{row.label}</span>
                    <p className="text-xs text-[var(--color-text-dim)]">{row.desc}</p>
                  </dt>
                  <dd className="whitespace-nowrap text-right font-mono text-[var(--color-text)]">
                    {row.value}
                  </dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
      </section>
    </div>
  );
}
