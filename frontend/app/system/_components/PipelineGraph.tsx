"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/ui/cn";
import type { PipelineNode, RoutingRule } from "@/lib/data/types";

const NODE_COLOR: Record<string, "indigo" | "emerald" | "amber" | "sky"> = {
  plan: "indigo",
  rewrite: "indigo",
  retrieve: "emerald",
  rerank: "amber",
  reason: "emerald",
  reflect: "sky",
};

const COLOR_VARS = {
  indigo: "var(--color-indigo)",
  emerald: "var(--color-emerald)",
  amber: "var(--color-amber)",
  sky: "var(--color-sky)",
} as const;

function Connector({ color }: { color: keyof typeof COLOR_VARS }) {
  return (
    <div className="relative h-px w-10 shrink-0 self-center sm:w-14">
      <div className="absolute inset-0 bg-gradient-to-r from-white/5 via-white/25 to-white/5" />
      <motion.div
        className="absolute top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full"
        style={{ backgroundColor: COLOR_VARS[color], boxShadow: `0 0 10px ${COLOR_VARS[color]}` }}
        animate={{ left: ["0%", "94%"] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}

const ROUTE_TONE: Record<string, { color: keyof typeof COLOR_VARS; label: string }> = {
  responder: { color: "emerald", label: "Emit response" },
  query_rewriter: { color: "amber", label: "Loop: rewrite" },
  reasoner: { color: "sky", label: "Loop: re-reason" },
  graceful_degradation: { color: "indigo", label: "Fallback" },
};

export function PipelineGraph({
  nodes,
  routingRules,
}: {
  nodes: PipelineNode[];
  routingRules: RoutingRule[];
}) {
  const [activeId, setActiveId] = useState(nodes[0]?.id ?? "");
  const active = nodes.find((n) => n.id === activeId) ?? nodes[0];

  return (
    <div>
      <div className="styled-scroll flex items-stretch overflow-x-auto pb-6">
        {nodes.map((node, i) => {
          const color = NODE_COLOR[node.id] ?? "indigo";
          const isActive = node.id === activeId;
          return (
            <div key={node.id} className="flex items-center">
              <motion.button
                onClick={() => setActiveId(node.id)}
                whileHover={{ y: -3 }}
                className={cn(
                  "glass-panel flex w-[168px] shrink-0 flex-col gap-2 p-4 text-left transition-all",
                  isActive ? "border-white/25" : "border-white/[0.06] opacity-70 hover:opacity-100",
                )}
                style={{
                  boxShadow: isActive
                    ? `inset 0 1px 0 0 rgba(255,255,255,0.08), 0 0 32px -8px ${COLOR_VARS[color]}`
                    : undefined,
                }}
              >
                <div className="flex items-center justify-between">
                  <span
                    className="font-mono text-[10px]"
                    style={{ color: COLOR_VARS[color] }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span
                    className={cn(
                      "h-1.5 w-1.5 rounded-full",
                      isActive && "pulse-dot",
                    )}
                    style={{
                      backgroundColor: COLOR_VARS[color],
                      "--pulse-color": `${COLOR_VARS[color]}88`,
                    } as React.CSSProperties}
                  />
                </div>
                <span className="text-sm font-semibold text-[var(--color-text)]">
                  {node.label}
                </span>
                <span className="text-xs leading-snug text-[var(--color-text-muted)]">
                  {node.desc}
                </span>
              </motion.button>
              {i < nodes.length - 1 && <Connector color={color} />}
            </div>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        {active && (
          <motion.div
            key={active.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="glass-panel grid gap-4 p-6 sm:grid-cols-4"
          >
            <div className="sm:col-span-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
                Model
              </span>
              <p className="mt-1 font-mono text-sm text-[var(--color-text)]">{active.model}</p>
            </div>
            <div className="sm:col-span-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
                Input
              </span>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{active.input}</p>
            </div>
            <div className="sm:col-span-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
                Output
              </span>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{active.output}</p>
            </div>
            <div className="sm:col-span-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
                Behavior
              </span>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{active.note}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-8">
        <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--color-text-dim)]">
          Reflector decision
        </span>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {routingRules.map((rule, i) => {
            const tone = ROUTE_TONE[rule.to] ?? { color: "indigo" as const, label: rule.to };
            return (
              <div
                key={i}
                className="glass-panel flex items-center justify-between gap-3 px-4 py-3"
              >
                <span className="font-mono text-xs text-[var(--color-text-muted)]">
                  {rule.condition}
                </span>
                <span
                  className="shrink-0 font-mono text-xs font-medium"
                  style={{ color: COLOR_VARS[tone.color] }}
                >
                  {tone.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
