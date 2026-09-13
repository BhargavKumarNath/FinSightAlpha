import { cn } from "./cn";

export function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6",
        "transition-colors hover:border-[var(--color-border-strong)]",
        className,
      )}
    >
      {children}
    </div>
  );
}
