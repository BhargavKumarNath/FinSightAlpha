import { cn } from "./cn";

const TONES = {
  neutral: "border-[var(--color-border-strong)] text-[var(--color-text-muted)]",
  gold: "border-[var(--color-gold)] text-[var(--color-gold)] bg-[var(--color-gold-dim)]",
  slate: "border-[var(--color-slate)] text-[var(--color-slate)] bg-[var(--color-slate-dim)]",
  caveat:
    "border-[var(--color-caveat)] text-[var(--color-caveat)] bg-[var(--color-caveat-dim)]",
} as const;

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: keyof typeof TONES;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-2 py-0.5 font-mono text-xs",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
