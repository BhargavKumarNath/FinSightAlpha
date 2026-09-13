import { cn } from "./cn";

/**
 * Measured | Illustrative | Live — the product's core honesty signal.
 * Every stat/chart in the app carries one of these. See deployment_roadmap.md §5.
 */
export type Provenance = "measured" | "illustrative" | "live";

const STYLES: Record<Provenance, string> = {
  measured:
    "border-[var(--color-gold)] text-[var(--color-gold)] bg-[var(--color-gold-dim)]",
  live: "border-[var(--color-slate)] text-[var(--color-slate)] bg-[var(--color-slate-dim)]",
  illustrative:
    "border-dashed border-[var(--color-text-dim)] text-[var(--color-text-muted)]",
};

const LABELS: Record<Provenance, string> = {
  measured: "Measured",
  live: "Live",
  illustrative: "Illustrative",
};

export function ProvenanceChip({
  kind,
  label,
  className,
}: {
  kind: Provenance;
  label?: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
        STYLES[kind],
        className,
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label ?? LABELS[kind]}
    </span>
  );
}
