import { getSystemData } from "@/lib/data/loaders";
import { SectionHeader, Card, Badge, ProvenanceChip } from "@/lib/ui";

const TIER_TONE = { GREEN: "gold", YELLOW: "neutral", RED: "caveat" } as const;

export default function SystemPage() {
  const data = getSystemData();
  const { pipeline, budget_tiers, model_router_table, cache_config, context_window } = data;

  return (
    <div className="mx-auto max-w-5xl px-4 py-16 sm:px-6">
      <SectionHeader
        eyebrow="System"
        title="How the agent actually works"
        description="The 7-node LangGraph pipeline, its conditional reflect → rewrite loop, and the budget/routing rules that govern it — read directly from the live optimization config, not hand-typed."
      />

      {/* Pipeline */}
      <section className="mb-16">
        <div className="mb-4 flex items-center gap-2">
          <h2 className="font-serif text-xl text-[var(--color-text)]">Pipeline</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {pipeline.nodes.map((node, i) => (
            <Card key={node.id} className="relative">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-[var(--color-gold)]">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <Badge tone="slate">{node.model}</Badge>
              </div>
              <h3 className="mt-2 font-serif text-lg text-[var(--color-text)]">
                {node.label}
              </h3>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">
                {node.desc}
              </p>
              <dl className="mt-3 space-y-1 border-t border-[var(--color-border)] pt-3 font-mono text-xs text-[var(--color-text-dim)]">
                <div className="flex justify-between gap-2">
                  <dt>in</dt>
                  <dd className="text-right text-[var(--color-text-muted)]">{node.input}</dd>
                </div>
                <div className="flex justify-between gap-2">
                  <dt>out</dt>
                  <dd className="text-right text-[var(--color-text-muted)]">{node.output}</dd>
                </div>
              </dl>
              <p className="mt-2 text-xs italic text-[var(--color-text-dim)]">{node.note}</p>
            </Card>
          ))}
        </div>

        <h3 className="mb-3 mt-8 font-serif text-lg text-[var(--color-text)]">
          Reflector routing
        </h3>
        <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
                <th className="px-4 py-2 font-normal">From</th>
                <th className="px-4 py-2 font-normal">Condition</th>
                <th className="px-4 py-2 font-normal">To</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {pipeline.routing_rules.map((rule, i) => (
                <tr key={i} className="border-b border-[var(--color-border)] last:border-0">
                  <td className="px-4 py-2 text-[var(--color-text)]">{rule.from}</td>
                  <td className="px-4 py-2 text-[var(--color-text-muted)]">{rule.condition}</td>
                  <td className="px-4 py-2 text-[var(--color-gold)]">{rule.to}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Budget tiers */}
      <section className="mb-16">
        <div className="mb-4 flex items-center gap-2">
          <h2 className="font-serif text-xl text-[var(--color-text)]">Budget tiers</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          {budget_tiers.map((tier) => (
            <Card key={tier.tier}>
              <Badge tone={TIER_TONE[tier.tier as keyof typeof TIER_TONE] ?? "neutral"}>
                {tier.tier}
              </Badge>
              <p className="mt-2 font-mono text-xs text-[var(--color-text-dim)]">
                {tier.range} tokens
              </p>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">{tier.desc}</p>
              <dl className="mt-3 space-y-1 border-t border-[var(--color-border)] pt-3 text-xs">
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">max loops</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.loops}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">retrieval top-n</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.top_n}</dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-[var(--color-text-dim)]">model</dt>
                  <dd className="font-mono text-[var(--color-text)]">{tier.model}</dd>
                </div>
              </dl>
              {tier.skips.length > 0 && (
                <p className="mt-2 text-xs text-[var(--color-caveat)]">
                  skips: {tier.skips.join(", ")}
                </p>
              )}
            </Card>
          ))}
        </div>
      </section>

      {/* Model router */}
      <section className="mb-16">
        <div className="mb-4 flex items-center gap-2">
          <h2 className="font-serif text-xl text-[var(--color-text)]">Model routing</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <div className="overflow-x-auto rounded-lg border border-[var(--color-border)]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border)] text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
                <th className="px-4 py-2 font-normal">Task</th>
                <th className="px-4 py-2 font-normal">Tier</th>
                <th className="px-4 py-2 font-normal">Model</th>
                <th className="px-4 py-2 font-normal">Reason</th>
                <th className="px-4 py-2 font-normal">Cost</th>
              </tr>
            </thead>
            <tbody>
              {model_router_table.map((row, i) => (
                <tr key={i} className="border-b border-[var(--color-border)] last:border-0">
                  <td className="px-4 py-2 text-[var(--color-text)]">{row.task}</td>
                  <td className="px-4 py-2 font-mono text-xs text-[var(--color-text-muted)]">{row.tier}</td>
                  <td className="px-4 py-2 font-mono text-xs text-[var(--color-gold)]">{row.model}</td>
                  <td className="px-4 py-2 text-[var(--color-text-muted)]">{row.reason}</td>
                  <td className="px-4 py-2 font-mono text-xs text-[var(--color-text-dim)]">{row.approx_cost}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Cache + context window */}
      <section className="grid gap-8 sm:grid-cols-2">
        <div>
          <div className="mb-4 flex items-center gap-2">
            <h2 className="font-serif text-xl text-[var(--color-text)]">Response cache</h2>
            <ProvenanceChip kind="measured" />
          </div>
          <Card>
            <dl className="space-y-2 text-sm">
              {cache_config.map((row) => (
                <div key={row.label} className="flex justify-between gap-4">
                  <dt className="text-[var(--color-text-muted)]">{row.label}</dt>
                  <dd className="text-right font-mono text-[var(--color-text)]">{row.value}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
        <div>
          <div className="mb-4 flex items-center gap-2">
            <h2 className="font-serif text-xl text-[var(--color-text)]">Context window</h2>
            <ProvenanceChip kind="measured" />
          </div>
          <Card>
            <dl className="space-y-3 text-sm">
              {context_window.map((row) => (
                <div key={row.label} className="flex justify-between gap-4">
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
