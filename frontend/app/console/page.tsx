import { getConsoleData } from "@/lib/data/loaders";
import { SectionHeader } from "@/lib/ui";
import { Terminal } from "./_components/Terminal";

export default function ConsolePage() {
  const { items } = getConsoleData();

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 sm:py-20">
      <SectionHeader
        eyebrow="Console"
        title="Query the engine directly."
        description="Execute a question and watch it move through planning, retrieval, reranking, reasoning, and reflection before an answer is emitted. Indexed queries below replay a genuine recorded execution end to end."
      />
      <Terminal replays={items} />
    </div>
  );
}
