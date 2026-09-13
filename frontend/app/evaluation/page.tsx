import { getEvaluationData } from "@/lib/data/loaders";
import { SectionHeader, Card, Stat, Badge, ProvenanceChip } from "@/lib/ui";

function MetricBar({ value, label }: { value: number; label: string }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <span className="font-mono text-[var(--color-text)]">{value.toFixed(2)}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-2)]">
        <div
          className="h-full rounded-full bg-[var(--color-gold)]"
          style={{ width: `${Math.max(value, 0.02) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function EvaluationPage() {
  const data = getEvaluationData();

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6">
      <SectionHeader
        eyebrow="Evaluation"
        title="A small, honest RAGAS pilot"
        description="Three real queries, scored for faithfulness and answer relevancy. No inflation, no resurrected radar charts — this is the eval that actually exists."
      />

      {!data.available || !data.summary ? (
        <Card>
          <p className="text-[var(--color-text-muted)]">
            No evaluation report found yet — run the RAGAS eval suite and
            re-run <code className="font-mono text-xs">scripts/precompute.py</code>.
          </p>
        </Card>
      ) : (
        <>
          {data.caveat && (
            <Card className="mb-8 border-[var(--color-caveat)]">
              <Badge tone="caveat">Caveat</Badge>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">{data.caveat}</p>
            </Card>
          )}

          <section className="mb-12 grid gap-6 sm:grid-cols-3">
            <Card>
              <Stat
                label="Queries evaluated"
                value={data.summary.num_queries}
                provenance="measured"
                description="3-query pilot, not a benchmark suite."
              />
            </Card>
            <Card>
              <Stat
                label="Avg. faithfulness"
                value={data.summary.avg_faithfulness.toFixed(3)}
                provenance="measured"
                description="Is the answer supported by retrieved context?"
              />
            </Card>
            <Card>
              <Stat
                label="Avg. answer relevancy"
                value={data.summary.avg_answer_relevancy.toFixed(3)}
                provenance="measured"
                description="Does the answer address the question asked?"
              />
            </Card>
          </section>

          <section>
            <div className="mb-4 flex items-center gap-2">
              <h2 className="font-serif text-xl text-[var(--color-text)]">Per-query results</h2>
              <ProvenanceChip kind="measured" />
            </div>
            <div className="space-y-4">
              {data.rows.map((row, i) => (
                <Card key={i}>
                  <p className="font-serif text-base text-[var(--color-text)]">{row.question}</p>
                  <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    <MetricBar value={row.faithfulness} label="Faithfulness" />
                    <MetricBar value={row.answer_relevancy} label="Answer relevancy" />
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
