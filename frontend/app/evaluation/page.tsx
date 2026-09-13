import { getEvaluationData } from "@/lib/data/loaders";
import { SectionHeader, Card, Badge, ProvenanceChip, RadialProgress, Sparkline } from "@/lib/ui";
import { MetricGauge } from "./_components/MetricGauge";

export default function EvaluationPage() {
  const data = getEvaluationData();

  return (
    <div className="mx-auto max-w-5xl px-4 py-16 sm:px-6 sm:py-20">
      <SectionHeader
        eyebrow="Evaluation"
        title="Faithfulness and relevancy, scored against ground truth."
        description="Every response is graded on whether its claims are actually supported by retrieved context, and whether it answers the question that was asked. These are live telemetry from evaluation runs, not projected targets."
      />

      {!data.available || !data.summary ? (
        <Card>
          <p className="text-[var(--color-text-muted)]">
            No evaluation telemetry is available yet. Run the evaluation
            suite to populate this view.
          </p>
        </Card>
      ) : (
        <>
          <section className="mb-12 grid gap-6 sm:grid-cols-[auto_auto_1fr] sm:items-center">
            <Card hover={false} className="flex flex-col items-center">
              <RadialProgress
                value={data.summary.avg_faithfulness}
                color="emerald"
                label="Faithfulness"
                sublabel="Claims supported by context"
              />
            </Card>
            <Card hover={false} className="flex flex-col items-center">
              <RadialProgress
                value={data.summary.avg_answer_relevancy}
                color="sky"
                label="Answer relevancy"
                sublabel="Response addresses the query"
              />
            </Card>
            <Card hover={false} className="flex h-full flex-col justify-between gap-4">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
                  Evaluation surface
                </span>
                <ProvenanceChip kind="measured" />
              </div>
              <div className="flex items-end justify-between gap-6">
                <div>
                  <p className="font-mono text-3xl font-medium text-[var(--color-text)]">
                    {data.summary.num_queries}
                  </p>
                  <p className="text-xs text-[var(--color-text-muted)]">queries scored</p>
                </div>
                <Sparkline
                  values={data.rows.map((r) => r.faithfulness)}
                  color="emerald"
                  width={120}
                  height={36}
                />
              </div>
              {data.caveat && (
                <p className="border-t border-white/[0.06] pt-3 text-xs leading-relaxed text-[var(--color-text-dim)]">
                  <Badge tone="amber" className="mr-1.5">
                    Methodology
                  </Badge>
                  {data.caveat}
                </p>
              )}
            </Card>
          </section>

          <section>
            <div className="mb-5 flex items-center gap-2">
              <h2 className="text-lg font-semibold text-[var(--color-text)]">
                Query level breakdown
              </h2>
              <ProvenanceChip kind="measured" />
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.rows.map((row, i) => (
                <Card key={i} glow="indigo" className="flex flex-col gap-4">
                  <div className="flex items-start justify-between gap-2">
                    <span className="font-mono text-[11px] text-[var(--color-text-dim)]">
                      Q{String(i + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <p className="line-clamp-3 text-sm text-[var(--color-text)]">
                    {row.question}
                  </p>
                  <div className="space-y-3 border-t border-white/[0.06] pt-4">
                    <MetricGauge label="Faithfulness" value={row.faithfulness} color="emerald" />
                    <MetricGauge label="Answer relevancy" value={row.answer_relevancy} color="sky" />
                  </div>
                </Card>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
