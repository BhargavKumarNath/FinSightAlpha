import { cn } from "./cn";

const TONES = {
  neutral: "border-white/10 text-[var(--color-text-muted)] bg-white/[0.03]",
  emerald:
    "border-[var(--color-emerald)]/30 text-[var(--color-emerald)] bg-[var(--color-emerald-dim)]",
  indigo:
    "border-[var(--color-indigo)]/30 text-[var(--color-indigo)] bg-[var(--color-indigo-dim)]",
  amber:
    "border-[var(--color-amber)]/30 text-[var(--color-amber)] bg-[var(--color-amber-dim)]",
  rose: "border-[var(--color-rose)]/30 text-[var(--color-rose)] bg-[var(--color-rose-dim)]",
  sky: "border-[var(--color-sky)]/30 text-[var(--color-sky)] bg-[var(--color-sky-dim)]",
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
        "inline-flex items-center rounded-md border px-2 py-0.5 font-mono text-[11px] tracking-wide",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
