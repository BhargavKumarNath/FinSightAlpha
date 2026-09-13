"use client";

import { motion } from "framer-motion";

export function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="mb-10 flex flex-col gap-3"
    >
      {eyebrow && (
        <div className="flex items-center gap-2">
          <span
            className="h-1.5 w-1.5 rounded-full bg-[var(--color-indigo)] pulse-dot"
            style={{ "--pulse-color": "rgba(129,140,248,0.55)" } as React.CSSProperties}
          />
          <span className="font-mono text-xs uppercase tracking-[0.2em] text-[var(--color-indigo)]">
            {eyebrow}
          </span>
        </div>
      )}
      <h1 className="text-3xl font-semibold tracking-tight text-[var(--color-text)] sm:text-4xl">
        {title}
      </h1>
      {description && (
        <p className="max-w-2xl text-[15px] leading-relaxed text-[var(--color-text-muted)]">
          {description}
        </p>
      )}
    </motion.div>
  );
}
