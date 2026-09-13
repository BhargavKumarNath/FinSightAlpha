"use client";

import { motion } from "framer-motion";

const COLOR = {
  emerald: "var(--color-emerald)",
  indigo: "var(--color-indigo)",
  amber: "var(--color-amber)",
  rose: "var(--color-rose)",
  sky: "var(--color-sky)",
} as const;

export function Sparkline({
  values,
  width = 160,
  height = 40,
  color = "indigo",
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: keyof typeof COLOR;
}) {
  if (values.length < 2) return null;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const step = width / (values.length - 1);

  const points = values.map((v, i) => {
    const x = i * step;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const path = `M${points.join(" L")}`;
  const areaPath = `${path} L${width},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`spark-fill-${color}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={COLOR[color]} stopOpacity={0.35} />
          <stop offset="100%" stopColor={COLOR[color]} stopOpacity={0} />
        </linearGradient>
      </defs>
      <motion.path
        d={areaPath}
        fill={`url(#spark-fill-${color})`}
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8, delay: 0.4 }}
      />
      <motion.path
        d={path}
        fill="none"
        stroke={COLOR[color]}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0 }}
        whileInView={{ pathLength: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
      />
    </svg>
  );
}
