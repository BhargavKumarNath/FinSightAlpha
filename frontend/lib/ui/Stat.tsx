import { ProvenanceChip, type Provenance } from "./ProvenanceChip";

export function Stat({
  label,
  value,
  unit,
  provenance,
  description,
}: {
  label: string;
  value: string | number;
  unit?: string;
  provenance?: Provenance;
  description?: string;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
          {label}
        </span>
        {provenance && <ProvenanceChip kind={provenance} />}
      </div>
      <div className="font-mono text-2xl text-[var(--color-text)]">
        {value}
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
