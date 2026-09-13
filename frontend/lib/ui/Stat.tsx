import { ProvenanceChip, type Provenance } from "./ProvenanceChip";
import { AnimatedNumber } from "./AnimatedNumber";

export function Stat({
  label,
  value,
  decimals = 0,
  unit,
  provenance,
  description,
  accent = "text",
}: {
  label: string;
  value: string | number;
  decimals?: number;
  unit?: string;
  provenance?: Provenance;
  description?: string;
  accent?: "text" | "emerald" | "indigo" | "amber";
}) {
  const ACCENT_CLASS = {
    text: "text-[var(--color-text)]",
    emerald: "text-[var(--color-emerald)]",
    indigo: "text-[var(--color-indigo)]",
    amber: "text-[var(--color-amber)]",
  } as const;
  const accentClass = ACCENT_CLASS[accent];

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
          {label}
        </span>
        {provenance && <ProvenanceChip kind={provenance} />}
      </div>
      <div className={`font-mono text-3xl font-medium tabular-nums ${accentClass}`}>
        {typeof value === "number" ? (
          <AnimatedNumber value={value} decimals={decimals} />
        ) : (
          value
        )}
        {unit && (
          <span className="ml-1.5 text-sm text-[var(--color-text-muted)]">
            {unit}
          </span>
        )}
      </div>
      {description && (
        <p className="text-sm text-[var(--color-text-muted)]">{description}</p>
      )}
    </div>
  );
}
