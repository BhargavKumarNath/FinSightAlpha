import Link from "next/link";
import { Card, Badge } from "@/lib/ui";

const DESTINATIONS = [
  {
    href: "/console",
    title: "Console",
    desc: "Ask the agent a question about the indexed filing and watch it reason.",
  },
  {
    href: "/system",
    title: "System",
    desc: "The 7-node LangGraph pipeline, budget tiers, and model routing — as configured.",
  },
  {
    href: "/evaluation",
    title: "Evaluation",
    desc: "The real RAGAS pilot run, shown at the scale it actually was.",
  },
  {
    href: "/corpus",
    title: "Corpus",
    desc: "What's actually indexed right now, and what filing types are supported.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-16 sm:px-6 sm:py-24">
      <span className="font-mono text-xs uppercase tracking-widest text-[var(--color-gold)]">
        Evidence-grounded research console
      </span>
      <h1 className="mt-3 max-w-2xl font-serif text-4xl leading-tight text-[var(--color-text)] sm:text-5xl">
        Every number here is labeled by what it actually is.
      </h1>
      <p className="mt-4 max-w-xl text-[var(--color-text-muted)]">
        FinSightAlpha is a small, honest RAG system over one real SEC filing.
        Nothing on this site is dressed up to look bigger than it is —
        <Badge tone="gold" className="mx-1">
          Measured
        </Badge>
        figures come from real runs,
        <Badge tone="slate" className="mx-1">
          Live
        </Badge>
        figures come from your own query, and anything
        <Badge tone="neutral" className="mx-1">
          Illustrative
        </Badge>
        is marked as such rather than hidden.
      </p>

      <div className="mt-12 grid gap-4 sm:grid-cols-2">
        {DESTINATIONS.map((d) => (
          <Link key={d.href} href={d.href} className="block">
            <Card className="h-full hover:border-[var(--color-gold)]">
              <h2 className="font-serif text-xl text-[var(--color-text)]">
                {d.title}
              </h2>
              <p className="mt-2 text-sm text-[var(--color-text-muted)]">
                {d.desc}
              </p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
