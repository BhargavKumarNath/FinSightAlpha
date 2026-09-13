export type Provenance = "measured" | "illustrative" | "live";

export interface PipelineNode {
  id: string;
  label: string;
  desc: string;
  model: string;
  prompt: string;
  input: string;
  output: string;
  note: string;
}

export interface RoutingRule {
  from: string;
  condition: string;
  to: string;
}

export interface BudgetTier {
  tier: string;
  range: string;
  loops: number;
  top_n: number;
  model: string;
  desc: string;
  skips: string[];
}

export interface ModelRouterRow {
  task: string;
  tier: string;
  model: string;
  reason: string;
  approx_cost: string;
}

export interface KeyValueRow {
  label: string;
  value: string;
}

export interface ContextWindowRow extends KeyValueRow {
  desc: string;
}

export interface SystemData {
  provenance: Provenance;
  pipeline: {
    nodes: PipelineNode[];
    routing_rules: RoutingRule[];
  };
  budget_tiers: BudgetTier[];
  model_router_table: ModelRouterRow[];
  cache_config: KeyValueRow[];
  context_window: ContextWindowRow[];
}

export interface EvaluationRow {
  question: string;
  faithfulness: number;
  answer_relevancy: number;
}

export interface EvaluationSummary {
  num_queries: number;
  avg_faithfulness: number;
  avg_answer_relevancy: number;
}

export interface EvaluationData {
  provenance: Provenance;
  available: boolean;
  rows: EvaluationRow[];
  summary: EvaluationSummary | null;
  caveat?: string;
}

export interface FilingItem {
  doc_id: string;
  title: string;
  source_name: string;
  content_hash: string;
  chunk_count: number;
  indexed_at: number;
  collection: string;
}

export interface ConsoleReplay {
  question: string;
  response: string;
  faithfulness: number;
  answer_relevancy: number;
  retrieved_count: number;
  cited_count: number;
}

export interface ConsoleData {
  provenance: Provenance;
  items: ConsoleReplay[];
}

export interface CorpusData {
  filings: {
    provenance: Provenance;
    filing_count: number;
    total_chunks: number;
    items: FilingItem[];
  };
  supported_filing_types: {
    provenance: Provenance;
    note: string;
    items: { type: string; desc: string }[];
  };
}

// Client-side types for the live chat pipeline (app/api/chat). Kept
// separate from lib/rag/types.ts, which is a server-only module tree.

export type ChatStage = "plan" | "rewrite" | "retrieve" | "rerank" | "reason" | "reflect" | "respond";

export type ChatStreamEvent =
  | { type: "stage"; stage: ChatStage; detail: string; ms: number }
  | {
      type: "result";
      question: string;
      response: string;
      retrieved_count: number;
      cited_count: number;
      plan: string;
      sub_queries: string[];
      loop_count: number;
      reflection: string;
      chunks: EvidenceChunk[];
    }
  | { type: "error"; message: string };

export interface EvidenceChunk {
  docId: number;
  text: string;
  source: string;
  score: number;
}

export interface EvidenceStageLog {
  stage: ChatStage;
  detail: string;
  ms: number;
}

export interface EvidenceRecord {
  queryId: string;
  question: string;
  response: string;
  plan: string;
  subQueries: string[];
  loopCount: number;
  reflection: string;
  retrievedCount: number;
  citedCount: number;
  chunks: EvidenceChunk[];
  stages: EvidenceStageLog[];
  createdAt: number;
}
