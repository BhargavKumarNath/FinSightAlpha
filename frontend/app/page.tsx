import Link from "next/link";
import { Card, Badge } from "@/lib/ui";

const DESTINATIONS = [
  {
    href: "/console",
    title: "Console",
    desc: "Query the reasoning engine directly and trace every retrieval, rerank, and synthesis step in real time.",
    tone: "indigo" as const,
  },
  {
    href: "/system",
    title: "System",
    desc: "The seven stage execution graph: planning, retrieval, reranking, and self correction, mapped end to end.",
    tone: "emerald" as const,
  },
  {
    href: "/evaluation",
    title: "Evaluation",
    desc: "Faithfulness and relevancy telemetry from live evaluation runs, scored against the source filing.",
    tone: "amber" as const,
  },
  {
    href: "/corpus",
    title: "Corpus",
    desc: "The indexed document register: hashes, chunk counts, ingestion timestamps, and coverage status.",
    tone: "sky" as const,
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-6xl px-4 pb-24 pt-20 sm:px-6 sm:pt-28">
      <div className="flex flex-col items-start gap-6">
        <Badge tone="indigo">Institutional grade retrieval engine</Badge>

        <h1 className="max-w-3xl text-4xl font-semibold leading-[1.1] tracking-tight text-[var(--color-text)] sm:text-6xl">
          Filing intelligence,{" "}
          <span className="shimmer-text">computed with rigor.</span>
        </h1>

        <p className="max-w-xl text-lg leading-relaxed text-[var(--color-text-muted)]">
          FinSight Alpha fuses hybrid retrieval, cross encoder reranking, and
          multi hop reasoning into a single execution pipeline over SEC
          filings. Every figure on this platform is graded by its own
          rigor: measured from a live run, indicative of expected shape,
          or streamed live from your own query.
        </p>

        <div className="flex flex-wrap items-center gap-4 pt-2 font-mono text-xs text-[var(--color-text-dim)]">
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-emerald)]" />
            Hybrid dense + sparse retrieval
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-indigo)]" />
            Reflective self correction
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-amber)]" />
            Citation grounded synthesis
          </span>
        </div>
      </div>

      <div className="mt-16 grid gap-4 sm:grid-cols-2">
        {DESTINATIONS.map((d) => (
          <Link key={d.href} href={d.href} className="block">
            <Card glow={d.tone} className="h-full">
              <div className="flex items-start justify-between">
                <h2 className="text-xl font-semibold text-[var(--color-text)]">
                  {d.title}
                </h2>
                <span className="font-mono text-[var(--color-text-dim)] transition-transform group-hover:translate-x-1">
                  &rarr;
                </span>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-[var(--color-text-muted)]">
                {d.desc}
              </p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
