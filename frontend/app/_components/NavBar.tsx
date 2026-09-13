"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/ui/cn";

const NAV_ITEMS = [
  { href: "/console", label: "Console" },
  { href: "/system", label: "System" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/corpus", label: "Corpus" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-10 border-b border-[var(--color-border)] bg-[var(--color-bg)]/90 backdrop-blur">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="shrink-0 font-serif text-lg text-[var(--color-text)]">
          FinSight<span className="text-[var(--color-gold)]">Alpha</span>
        </Link>

        <nav className="flex items-center gap-1 overflow-x-auto">
          {NAV_ITEMS.map((item) => {
            const active =
              pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "shrink-0 rounded px-3 py-1.5 text-sm transition-colors",
                  active
                    ? "bg-[var(--color-surface-2)] text-[var(--color-text)]"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text)]",
                )}
              >
                {item.label}
              </Link>
            );
          })}
          <span
            className="shrink-0 cursor-not-allowed rounded px-3 py-1.5 text-sm text-[var(--color-text-dim)]"
            title="Opens after your first Console query"
          >
            Evidence Trail
          </span>
        </nav>

        <div
          className="hidden shrink-0 items-center gap-1.5 rounded-full border border-dashed border-[var(--color-text-dim)] px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-muted)] sm:inline-flex"
          title="The serverless chat pipeline (Qdrant + Groq) is built in Phase 2"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-text-dim)]" />
          Chat backend: Phase 2
        </div>
      </div>
    </header>
  );
}
