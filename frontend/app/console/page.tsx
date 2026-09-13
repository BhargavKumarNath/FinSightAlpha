import { SectionHeader, Card, Badge } from "@/lib/ui";

export default function ConsolePage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <SectionHeader
        eyebrow="Console"
        title="Live Q&A is coming in Phase 2"
        description="This page will host the streaming chat experience — plan, retrieve, rerank, reason, reflect — backed by Qdrant Cloud and Groq. It isn't wired up yet, so nothing here pretends to be live."
      />
      <Card>
        <Badge tone="neutral">Not yet connected</Badge>
        <p className="mt-3 text-sm text-[var(--color-text-muted)]">
          Phase 0–1 shipped the design system and the static System,
          Evaluation, and Corpus pages. The serverless retrieval pipeline
          (ONNX embeddings, Qdrant Cloud, BM25 fusion, Groq reasoning loop)
          lands in Phase 2 per{" "}
          <code className="font-mono text-xs">deployment_roadmap.md</code>.
        </p>
      </Card>
    </div>
  );
}
