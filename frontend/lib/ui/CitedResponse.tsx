import { cn } from "./cn";

const CITATION_RE = /\[Doc\s*\d+[^\]]*\]/g;
const BOLD_RE = /\*\*(.+?)\*\*/g;

/** Renders **bold** markers from model output as actual bold text. */
function renderBold(text: string, keyPrefix: string) {
  const segments = text.split(BOLD_RE);
  return segments.map((seg, i) =>
    i % 2 === 1 ? (
      <strong key={`${keyPrefix}-b${i}`} className="font-semibold text-[var(--color-text)]">
        {seg}
      </strong>
    ) : (
      seg
    ),
  );
}

/**
 * Renders model output with inline [Doc N: ...] citation markers pulled out
 * as visible chips, so the grounding is legible at a glance rather than
 * buried in prose.
 */
export function CitedResponse({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const parts = text.split(CITATION_RE);
  const citations = text.match(CITATION_RE) ?? [];

  return (
    <p className={cn("text-[var(--color-text)]", className)}>
      {parts.map((part, i) => (
        <span key={i}>
          {renderBold(part, `p${i}`)}
          {citations[i] && (
            <span className="mx-1 inline-flex items-center rounded border border-[var(--color-indigo)]/30 bg-[var(--color-indigo-dim)] px-1.5 py-0.5 align-middle font-mono text-[11px] text-[var(--color-indigo)]">
              {citations[i].replace(/[[\]]/g, "")}
            </span>
          )}
        </span>
      ))}
    </p>
  );
}
