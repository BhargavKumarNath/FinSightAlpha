import Link from "next/link";
import { getEvaluationData, getCorpusData, getConsoleData } from "@/lib/data/loaders";
import { Card, Badge, ProvenanceChip, Stat, CitedResponse } from "@/lib/ui";

const DESTINATIONS = [
  {
    href: "/console",
    title: "Console",
    desc: "Run a query and read the answer it produces, citations included.",
    tone: "indigo" as const,
  },
  {
    href: "/system",
    title: "System",
    desc: "The execution graph behind every answer, for when you want the mechanics.",
    tone: "emerald" as const,
  },
  {
    href: "/evaluation",
    title: "Evaluation",
    desc: "Every scored run, with the answer that earned each score.",
    tone: "amber" as const,
  },
  {
    href: "/corpus",
    title: "Corpus",
    desc: "The filings currently indexed and searchable.",
    tone: "sky" as const,
  },
];

export default function Home() {
  const evaluation = getEvaluationData();
  const corpus = getCorpusData();
  const { items: replays } = getConsoleData();
  const featured = replays[0];

  return (
    <div className="mx-auto max-w-6xl px-4 pb-24 pt-20 sm:px-6 sm:pt-28">
      <div className="flex flex-col items-start gap-6">
        <Badge tone="indigo">Institutional grade retrieval engine</Badge>

        <h1 className="max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight text-[var(--color-text)] sm:text-6xl">
          Real answers, <span className="shimmer-text">grounded in real filings.</span>
        </h1>

        <p className="max-w-xl text-lg leading-relaxed text-[var(--color-text-muted)]">
          FinSight Alpha reads a filing, answers a question, and shows its
          work: every claim traces back to a specific passage. Below is an
          actual run, not a mockup.
        </p>
      </div>

      {featured && (
        <section className="mt-14">
          <div className="mb-4 flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-[var(--color-text-dim)]">
              Sample output
            </span>
            <ProvenanceChip kind="measured" />
          </div>
          <Card glow="indigo" hover={false} className="p-0 overflow-hidden">
            <div className="border-b border-white/[0.06] px-6 py-4">
              <p className="text-sm text-[var(--color-text-muted)]">Query</p>
              <p className="mt-1 text-base font-medium text-[var(--color-text)]">
                {featured.question}
              </p>
            </div>
            <div className="px-6 py-5">
              <p className="text-sm text-[var(--color-text-muted)]">Answer</p>
              <CitedResponse text={featured.response} className="mt-2 text-[15px] leading-relaxed" />
            </div>
            <div className="flex flex-wrap gap-6 border-t border-white/[0.06] px-6 py-4 font-mono text-xs text-[var(--color-text-muted)]">
              <span>
                Faithfulness <b className="text-[var(--color-emerald)]">{featured.faithfulness.toFixed(3)}</b>
              </span>
              <span>
                Passages cited <b className="text-[var(--color-text)]">{featured.cited_count}</b> of {featured.retrieved_count} retrieved
              </span>
            </div>
          </Card>
        </section>
      )}

      <section className="mt-14 grid gap-4 sm:grid-cols-4">
        <Card>
          <Stat label="Faithfulness, avg" value={evaluation.summary?.avg_faithfulness ?? 0} decimals={3} provenance="measured" accent="emerald" />
        </Card>
        <Card>
          <Stat label="Relevancy, avg" value={evaluation.summary?.avg_answer_relevancy ?? 0} decimals={3} provenance="measured" accent="indigo" />
        </Card>
        <Card>
          <Stat label="Chunks indexed" value={corpus.filings.total_chunks} provenance="measured" />
        </Card>
        <Card>
          <Stat label="Filings covered" value={corpus.filings.filing_count} provenance="measured" />
        </Card>
      </section>

      <div className="mt-16 grid gap-4 sm:grid-cols-2">
        {DESTINATIONS.map((d) => (
          <Link key={d.href} href={d.href} className="block">
            <Card glow={d.tone} className="h-full" hover>
              <div className="flex items-start justify-between">
                <h2 className="text-lg font-semibold text-[var(--color-text)]">
                  {d.title}
                </h2>
                <span className="font-mono text-[var(--color-text-dim)]">&rarr;</span>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-[var(--color-text-muted)]">
                {d.desc}
              </p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
