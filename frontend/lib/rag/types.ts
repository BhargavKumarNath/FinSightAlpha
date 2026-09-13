export type IndexedChunk = {
  id: number;
  text: string;
  source_name: string;
  chunk_index: number | null;
  content_hash: string | null;
  vector: number[];
};

export type VectorIndexFile = {
  provenance: string;
  model: string;
  dimensions: number;
  chunks: IndexedChunk[];
};

export type RetrievedChunk = {
  docId: number;
  text: string;
  source: string;
  score: number;
};

export type StageId =
  | "plan"
  | "rewrite"
  | "retrieve"
  | "rerank"
  | "reason"
  | "reflect"
  | "respond";

export type StageEvent = {
  type: "stage";
  stage: StageId;
  detail: string;
  ms: number;
};

export type ResultEvent = {
  type: "result";
  question: string;
  response: string;
  retrieved_count: number;
  cited_count: number;
  plan: string;
  sub_queries: string[];
  loop_count: number;
  reflection: string;
  chunks: RetrievedChunk[];
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type ChatEvent = StageEvent | ResultEvent | ErrorEvent;
