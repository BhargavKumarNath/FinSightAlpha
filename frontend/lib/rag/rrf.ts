/**
 * Reciprocal Rank Fusion, matching HybridRetriever.search()'s fusion step
 * in src/retrieval/hybrid_retriever.py: score(id) = sum(1 / (k + rank + 1))
 * over every ranked list the id appears in, ranks 0-indexed.
 */

const RRF_K = 60;

export function reciprocalRankFusion<T>(
  denseRanked: T[],
  sparseRanked: T[],
  topN: number,
): { id: T; rrfScore: number }[] {
  const scores = new Map<T, number>();
  denseRanked.forEach((id, rank) => {
    scores.set(id, (scores.get(id) ?? 0) + 1 / (RRF_K + rank + 1));
  });
  sparseRanked.forEach((id, rank) => {
    scores.set(id, (scores.get(id) ?? 0) + 1 / (RRF_K + rank + 1));
  });
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([id, rrfScore]) => ({ id, rrfScore }));
}
