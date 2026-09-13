"use client";

import { useState } from "react";
import { Card, CitedResponse } from "@/lib/ui";
import { MetricGauge } from "./MetricGauge";

export function QueryResultCard({
  index,
  question,
  response,
  faithfulness,
  answerRelevancy,
}: {
  index: number;
  question: string;
  response?: string;
  faithfulness: number;
  answerRelevancy: number;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Card glow="indigo" className="flex flex-col gap-4">
      <span className="font-mono text-[11px] text-[var(--color-text-dim)]">
        Q{String(index + 1).padStart(2, "0")}
      </span>
      <p className="text-sm font-medium text-[var(--color-text)]">{question}</p>

      <div className="space-y-3 border-t border-white/[0.06] pt-4">
        <MetricGauge label="Faithfulness" value={faithfulness} color="emerald" />
        <MetricGauge label="Answer relevancy" value={answerRelevancy} color="sky" />
      </div>

      {response && (
        <div className="border-t border-white/[0.06] pt-4">
          <button
            onClick={() => setOpen((v) => !v)}
            className="font-mono text-xs text-[var(--color-indigo)] hover:text-[var(--color-text)]"
          >
            {open ? "Hide the answer" : "Show the answer this scored"}
          </button>
          {open && (
            <CitedResponse
              text={response}
              className="mt-3 text-[13px] leading-relaxed text-[var(--color-text-muted)]"
            />
          )}
        </div>
      )}
    </Card>
  );
}
