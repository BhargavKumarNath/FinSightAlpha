import { getCorpusData } from "@/lib/data/loaders";
import { SectionHeader, Card, Stat, Badge, ProvenanceChip } from "@/lib/ui";
import { CorpusTable } from "./_components/CorpusTable";

export default function CorpusPage() {
  const { filings, supported_filing_types } = getCorpusData();
  const indexedTypes = new Set(["10-K (Annual)"]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-16 sm:px-6 sm:py-20">
      <SectionHeader
        eyebrow="Corpus"
        title="The document register."
        description="Every filing indexed into the retrieval layer, down to its content hash and chunk count. This is the ground truth the engine actually reasons against."
      />

      <section className="mb-12 grid gap-6 sm:grid-cols-2">
        <Card glow="emerald">
          <Stat label="Filings indexed" value={filings.filing_count} provenance="measured" />
        </Card>
        <Card glow="indigo">
          <Stat label="Total chunks" value={filings.total_chunks} provenance="measured" />
        </Card>
      </section>

      <section className="mb-14">
        <div className="mb-5 flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--color-text)]">Registry</h2>
          <ProvenanceChip kind="measured" />
        </div>
        <CorpusTable items={filings.items} />
      </section>

      <section>
        <div className="mb-5 flex items-center gap-2">
          <h2 className="text-lg font-semibold text-[var(--color-text)]">Filing type coverage</h2>
          <ProvenanceChip kind="illustrative" />
        </div>
        <p className="mb-5 text-sm text-[var(--color-text-muted)]">{supported_filing_types.note}</p>
        <div className="grid gap-4 sm:grid-cols-2">
          {supported_filing_types.items.map((ft) => {
            const isLive = indexedTypes.has(ft.type);
            return (
              <Card key={ft.type} glow={isLive ? "emerald" : "none"}>
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold text-[var(--color-text)]">{ft.type}</h3>
                  <Badge tone={isLive ? "emerald" : "neutral"}>
                    {isLive ? "Indexed" : "Parser ready"}
                  </Badge>
                </div>
                <p className="mt-2 text-sm text-[var(--color-text-muted)]">{ft.desc}</p>
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
}
