"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/ui/cn";
import { getLatestEvidenceId, onEvidenceUpdated } from "@/lib/evidence/store";

const NAV_ITEMS = [
  { href: "/console", label: "Console" },
  { href: "/system", label: "System" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/corpus", label: "Corpus" },
];

export function NavBar() {
  const pathname = usePathname();
  const [latestEvidenceId, setLatestEvidenceId] = useState<string | null>(null);

  useEffect(() => {
    const id = requestAnimationFrame(() => setLatestEvidenceId(getLatestEvidenceId()));
    const unsubscribe = onEvidenceUpdated(() => setLatestEvidenceId(getLatestEvidenceId()));
    return () => {
      cancelAnimationFrame(id);
      unsubscribe();
    };
  }, []);

  return (
    <header className="sticky top-0 z-20 border-b border-white/[0.06] bg-[#090a0f]/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
        <Link href="/" className="group flex shrink-0 items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-[var(--color-indigo)] to-[var(--color-sky)] font-mono text-[11px] font-bold text-[#090a0f]">
            F
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-[var(--color-text)]">
            FinSight <span className="text-[var(--color-indigo)]">Alpha</span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 overflow-x-auto rounded-full border border-white/[0.06] bg-white/[0.02] p-1">
          {NAV_ITEMS.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "relative shrink-0 rounded-full px-3.5 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-white/[0.08] text-[var(--color-text)]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
                )}
              >
                {item.label}
              </Link>
            );
          })}
          {latestEvidenceId ? (
            <Link
              href={`/evidence/${latestEvidenceId}`}
              className={cn(
                "relative hidden shrink-0 rounded-full px-3.5 py-1.5 text-sm transition-colors sm:inline-block",
                pathname.startsWith("/evidence")
                  ? "bg-white/[0.08] text-[var(--color-text)]"
                  : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
              )}
            >
              Evidence Trail
            </Link>
          ) : (
            <span
              className="hidden shrink-0 cursor-not-allowed rounded-full px-3.5 py-1.5 text-sm text-[var(--color-text-dim)] sm:inline-block"
              title="Unlocks after your first live Console query"
            >
              Evidence Trail
            </span>
          )}
        </nav>

        <div
          className="hidden shrink-0 items-center gap-2 rounded-full border border-[var(--color-emerald)]/25 bg-[var(--color-emerald-dim)] px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-[var(--color-emerald)] sm:inline-flex"
          title="Retrieval and reasoning pipeline is live"
        >
          <span
            className="h-1.5 w-1.5 rounded-full bg-[var(--color-emerald)] pulse-dot"
            style={{ "--pulse-color": "rgba(52,211,153,0.55)" } as React.CSSProperties}
          />
          Engine: Live
        </div>
      </div>
    </header>
  );
}
