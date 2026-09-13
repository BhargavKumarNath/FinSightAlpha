import { cn } from "./cn";

/**
 * Measured | Indicative | Live: the rigor signal carried on every figure
 * in the product. "Indicative" covers figures with no live/measured run
 * behind them yet (the standard finance term for non-firm data) rather
 * than a bare "mock" label.
 */
export type Provenance = "measured" | "illustrative" | "live";

const STYLES: Record<Provenance, string> = {
  measured:
    "border-[var(--color-emerald)]/35 text-[var(--color-emerald)] bg-[var(--color-emerald-dim)]",
  live: "border-[var(--color-sky)]/35 text-[var(--color-sky)] bg-[var(--color-sky-dim)]",
  illustrative:
    "border-dashed border-white/15 text-[var(--color-text-muted)] bg-white/[0.02]",
};

const LABELS: Record<Provenance, string> = {
  measured: "Measured",
  live: "Live",
  illustrative: "Indicative",
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
        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider",
        STYLES[kind],
        className,
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full bg-current",
          kind === "live" && "pulse-dot",
        )}
        style={
          kind === "live"
            ? ({ "--pulse-color": "rgba(56,189,248,0.55)" } as React.CSSProperties)
            : undefined
        }
      />
      {label ?? LABELS[kind]}
    </span>
  );
}
