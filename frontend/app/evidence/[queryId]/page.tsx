import { EvidenceView } from "./_components/EvidenceView";

export default async function EvidencePage({ params }: { params: Promise<{ queryId: string }> }) {
  const { queryId } = await params;
  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 sm:py-20">
      <EvidenceView queryId={queryId} />
    </div>
  );
}
