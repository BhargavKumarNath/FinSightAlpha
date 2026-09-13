"use client";

import { motion } from "framer-motion";

const COLOR = {
  emerald: "var(--color-emerald)",
  indigo: "var(--color-indigo)",
  amber: "var(--color-amber)",
  rose: "var(--color-rose)",
  sky: "var(--color-sky)",
} as const;

export function RadialProgress({
  value,
  size = 128,
  strokeWidth = 8,
  color = "emerald",
  label,
  sublabel,
}: {
  value: number; // 0..1
  size?: number;
  strokeWidth?: number;
  color?: keyof typeof COLOR;
  label: string;
  sublabel?: string;
}) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(1, value));

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.07)"
            strokeWidth={strokeWidth}
          />
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={COLOR[color]}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            whileInView={{ strokeDashoffset: circumference * (1 - clamped) }}
            viewport={{ once: true }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
            style={{
              filter: `drop-shadow(0 0 6px ${COLOR[color]}66)`,
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-xl font-medium text-[var(--color-text)]">
            {(clamped * 100).toFixed(0)}
          </span>
          <span className="font-mono text-[10px] text-[var(--color-text-dim)]">
            / 100
          </span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-[var(--color-text)]">{label}</p>
        {sublabel && (
          <p className="text-xs text-[var(--color-text-muted)]">{sublabel}</p>
        )}
      </div>
    </div>
  );
}
