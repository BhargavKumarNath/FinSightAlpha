import { getCorpusData } from "@/lib/data/loaders";
import { SectionHeader, Card, Stat, Badge, ProvenanceChip } from "@/lib/ui";

function formatIndexedAt(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export default function CorpusPage() {
  const { filings, supported_filing_types } = getCorpusData();

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6">
      <SectionHeader
        eyebrow="Corpus"
        title="What's actually indexed"
        description="One filing today. The registry below is the real Qdrant ingestion record, not a placeholder — this is a dataset that's meant to grow, not a generic pipeline diagram."
      />

      <section className="mb-12 grid gap-6 sm:grid-cols-2">
        <Card>
          <Stat
            label="Filings indexed"
            value={filings.filing_count}
            provenance="measured"
          />
        </Card>
        <Card>
          <Stat
            label="Total chunks"
            value={filings.total_chunks}
            provenance="measured"
          />
        </Card>
      </section>

      <section className="mb-12">
        <div className="mb-4 flex items-center gap-2">
          <h2 className="font-serif text-xl text-[var(--color-text)]">Registry</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <div className="space-y-3">
          {filings.items.map((f) => (
            <Card key={f.doc_id}>
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h3 className="font-serif text-lg text-[var(--color-text)]">{f.title}</h3>
                  <p className="text-sm text-[var(--color-text-muted)]">{f.source_name}</p>
                </div>
                <Badge tone="gold">{f.chunk_count} chunks</Badge>
              </div>
              <dl className="mt-4 grid gap-2 border-t border-[var(--color-border)] pt-4 text-xs sm:grid-cols-3">
                <div>
                  <dt className="text-[var(--color-text-dim)]">doc id</dt>
                  <dd className="font-mono text-[var(--color-text-muted)]">{f.doc_id}</dd>
                </div>
                <div>
                  <dt className="text-[var(--color-text-dim)]">content hash</dt>
                  <dd className="font-mono text-[var(--color-text-muted)]">{f.content_hash}</dd>
                </div>
                <div>
                  <dt className="text-[var(--color-text-dim)]">indexed</dt>
                  <dd className="font-mono text-[var(--color-text-muted)]">
                    {formatIndexedAt(f.indexed_at)}
                  </dd>
                </div>
              </dl>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-4 flex items-center gap-2">
          <h2 className="font-serif text-xl text-[var(--color-text)]">
            Filing types the parser supports
          </h2>
          <ProvenanceChip kind="illustrative" />
        </div>
        <p className="mb-4 text-sm text-[var(--color-text-muted)]">
          {supported_filing_types.note}
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          {supported_filing_types.items.map((ft) => (
            <Card key={ft.type}>
              <h3 className="font-serif text-base text-[var(--color-text)]">{ft.type}</h3>
              <p className="mt-1 text-sm text-[var(--color-text-muted)]">{ft.desc}</p>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
