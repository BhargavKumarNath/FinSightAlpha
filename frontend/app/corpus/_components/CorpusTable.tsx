"use client";

import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "@/lib/ui";
import type { FilingItem } from "@/lib/data/types";

function formatIndexedAt(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function HashChip({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        } catch {
          // Clipboard access denied; chip stays inert.
        }
      }}
      className="group inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/[0.03] px-2 py-1 font-mono text-[11px] text-[var(--color-text-muted)] transition-colors hover:border-white/20 hover:text-[var(--color-text)]"
      title="Copy content hash"
    >
      {value}
      <span className="text-[var(--color-text-dim)] group-hover:text-[var(--color-indigo)]">
        {copied ? "✓" : "⧉"}
      </span>
    </button>
  );
}

export function CorpusTable({ items }: { items: FilingItem[] }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (f) =>
        f.title.toLowerCase().includes(q) ||
        f.source_name.toLowerCase().includes(q) ||
        f.doc_id.toLowerCase().includes(q),
    );
  }, [items, query]);

  return (
    <div className="glass-panel overflow-hidden p-0">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.06] px-5 py-4">
        <div className="relative">
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-dim)]">
            ⌕
          </span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter by title, source, or document id"
            className="w-64 rounded-lg border border-white/10 bg-white/[0.03] py-2 pl-8 pr-3 text-sm text-[var(--color-text)] placeholder:text-[var(--color-text-dim)] focus:border-[var(--color-indigo)]/50 focus:outline-none"
          />
        </div>
        <span className="font-mono text-xs text-[var(--color-text-dim)]">
          {filtered.length} of {items.length} document{items.length === 1 ? "" : "s"}
        </span>
      </div>

      <div className="styled-scroll overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="border-b border-white/[0.06] text-xs uppercase tracking-wider text-[var(--color-text-dim)]">
              <th className="px-5 py-3 font-normal">Document</th>
              <th className="px-5 py-3 font-normal">Content hash</th>
              <th className="px-5 py-3 font-normal">Chunks</th>
              <th className="px-5 py-3 font-normal">Indexed</th>
              <th className="px-5 py-3 font-normal">Status</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence>
              {filtered.map((f) => (
                <motion.tr
                  key={f.doc_id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="border-b border-white/[0.03] last:border-0 hover:bg-white/[0.02]"
                >
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--color-emerald-dim)] font-mono text-[10px] font-semibold text-[var(--color-emerald)]">
                        10-K
                      </span>
                      <div>
                        <p className="font-medium text-[var(--color-text)]">{f.title}</p>
                        <p className="text-xs text-[var(--color-text-dim)]">{f.source_name}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4">
                    <HashChip value={f.content_hash} />
                  </td>
                  <td className="px-5 py-4 font-mono text-[var(--color-text)]">
                    {f.chunk_count.toLocaleString()}
                  </td>
                  <td className="px-5 py-4 font-mono text-xs text-[var(--color-text-muted)]">
                    {formatIndexedAt(f.indexed_at)}
                  </td>
                  <td className="px-5 py-4">
                    <Badge tone="emerald">
                      <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-current" />
                      Indexed
                    </Badge>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="px-5 py-8 text-center text-sm text-[var(--color-text-dim)]">
                  No documents match that filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
