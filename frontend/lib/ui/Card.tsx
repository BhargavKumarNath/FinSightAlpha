"use client";

import { motion, type HTMLMotionProps } from "framer-motion";
import { cn } from "./cn";

const GLOW = {
  none: "",
  emerald: "hover:shadow-[0_0_40px_-12px_var(--color-emerald)]",
  indigo: "hover:shadow-[0_0_40px_-12px_var(--color-indigo)]",
  amber: "hover:shadow-[0_0_40px_-12px_var(--color-amber)]",
  rose: "hover:shadow-[0_0_40px_-12px_var(--color-rose)]",
  sky: "hover:shadow-[0_0_40px_-12px_var(--color-sky)]",
} as const;

export function Card({
  children,
  className,
  glow = "none",
  hover = true,
  ...motionProps
}: {
  children: React.ReactNode;
  className?: string;
  glow?: keyof typeof GLOW;
  hover?: boolean;
} & HTMLMotionProps<"div">) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "glass-panel p-6",
        hover && "glass-panel-hover",
        hover && GLOW[glow],
        className,
      )}
      {...motionProps}
    >
      {children}
    </motion.div>
  );
}
