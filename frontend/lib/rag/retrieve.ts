import "server-only";
import { getVectorIndex } from "./vectorIndex";
import { buildBm25Index, bm25Scores, tokenize, type Bm25Index } from "./bm25";
import { embedQuery } from "./embed";
import { rerankScores } from "./rerank";
import { reciprocalRankFusion as rrfFuse } from "./rrf";
import type { IndexedChunk } from "./types";

/**
 * Hybrid retrieve + rerank, ported from HybridRetriever.search() in
 * src/retrieval/hybrid_retriever.py: dense (cosine) + sparse (BM25) fused
 * with RRF, then cross-encoder reranked. The LRU/TTL caching layers in the
 * Python version are session-scoped optimizations, not correctness --
 * skipped here (see deployment_roadmap.md §9, Phase 2 deviations).
 */

let bm25Singleton: { index: Bm25Index; ids: number[] } | null = null;

function getBm25(): { index: Bm25Index; ids: number[] } {
  if (!bm25Singleton) {
    const { chunks } = getVectorIndex();
    bm25Singleton = {
      index: buildBm25Index(chunks.map((c) => c.text)),
      ids: chunks.map((c) => c.id),
    };
  }
  return bm25Singleton;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function reciprocalRankFusion(
  denseRanked: number[],
  sparseRanked: number[],
  chunksById: Map<number, IndexedChunk>,
  topN: number,
): { id: number; chunk: IndexedChunk; rrfScore: number }[] {
  return rrfFuse(denseRanked, sparseRanked, topN).map(({ id, rrfScore }) => ({
    id,
    chunk: chunksById.get(id)!,
    rrfScore,
  }));
}

export type ScoredChunk = { id: number; text: string; source: string; score: number };

export async function retrieve(query: string, topN = 8, fetchK = 50): Promise<ScoredChunk[]> {
  const { chunks } = getVectorIndex();
  const chunksById = new Map(chunks.map((c) => [c.id, c]));

  const queryVector = await embedQuery(query);
  const denseRanked = chunks
    .map((c) => ({ id: c.id, score: cosineSimilarity(queryVector, c.vector) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, fetchK)
    .map((r) => r.id);

  const { index: bm25, ids: bm25Ids } = getBm25();
  const sparseScores = bm25Scores(bm25, tokenize(query));
  const sparseRanked = bm25Ids
    .map((id, i) => ({ id, score: sparseScores[i] }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, fetchK)
    .map((r) => r.id);

  const fused = reciprocalRankFusion(denseRanked, sparseRanked, chunksById, Math.max(topN * 2, 20));
  if (fused.length === 0) return [];

  const rerankInput = fused.map((f) => f.chunk.text);
  const rerankOut = await rerankScores(query, rerankInput);

  return fused
    .map((f, i) => ({
      id: f.id,
      text: f.chunk.text,
      source: f.chunk.source_name,
      score: rerankOut[i],
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}
