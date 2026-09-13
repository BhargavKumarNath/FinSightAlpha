"use client";

import { motion } from "framer-motion";

const COLOR = {
  emerald: "var(--color-emerald)",
  amber: "var(--color-amber)",
  sky: "var(--color-sky)",
  rose: "var(--color-rose)",
} as const;

export function MetricGauge({
  label,
  value,
  color = "emerald",
}: {
  label: string;
  value: number;
  color?: keyof typeof COLOR;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="text-[var(--color-text-muted)]">{label}</span>
        <span className="font-mono font-medium text-[var(--color-text)]">
          {value.toFixed(3)}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.05]">
        <motion.div
          className="h-full rounded-full"
          style={{
            background: `linear-gradient(90deg, ${COLOR[color]}88, ${COLOR[color]})`,
            boxShadow: `0 0 12px -2px ${COLOR[color]}`,
          }}
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(value, 0.015) * 100}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  );
}
