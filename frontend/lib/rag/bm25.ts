/**
 * BM25Okapi, ported term-for-term from rank_bm25 (the library the Python
 * backend uses via src/retrieval/hybrid_retriever.py), so sparse scores
 * here match the production ranking behavior rather than approximating it.
 */

export type Bm25Index = {
  size: number;
  avgdl: number;
  docLen: number[];
  docFreqs: Map<string, number>[];
  idf: Map<string, number>;
  k1: number;
  b: number;
};

export function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\s+/).filter(Boolean);
}

export function buildBm25Index(docs: string[], k1 = 1.5, b = 0.75, epsilon = 0.25): Bm25Index {
  const size = docs.length;
  const tokenizedDocs = docs.map(tokenize);
  const docLen = tokenizedDocs.map((d) => d.length);
  const avgdl = docLen.reduce((sum, len) => sum + len, 0) / size;

  const docFreqs: Map<string, number>[] = tokenizedDocs.map((tokens) => {
    const freq = new Map<string, number>();
    for (const t of tokens) freq.set(t, (freq.get(t) ?? 0) + 1);
    return freq;
  });

  const docsContaining = new Map<string, number>();
  for (const freq of docFreqs) {
    for (const word of freq.keys()) {
      docsContaining.set(word, (docsContaining.get(word) ?? 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  let idfSum = 0;
  const negativeIdfWords: string[] = [];
  for (const [word, freq] of docsContaining) {
    const value = Math.log(size - freq + 0.5) - Math.log(freq + 0.5);
    idf.set(word, value);
    idfSum += value;
    if (value < 0) negativeIdfWords.push(word);
  }
  const averageIdf = idf.size > 0 ? idfSum / idf.size : 0;
  const eps = epsilon * averageIdf;
  for (const word of negativeIdfWords) idf.set(word, eps);

  return { size, avgdl, docLen, docFreqs, idf, k1, b };
}

export function bm25Scores(index: Bm25Index, queryTokens: string[]): number[] {
  const scores = new Array(index.size).fill(0);
  for (const term of queryTokens) {
    const idfValue = index.idf.get(term);
    if (!idfValue) continue;
    for (let i = 0; i < index.size; i++) {
      const freq = index.docFreqs[i].get(term) ?? 0;
      if (freq === 0) continue;
      const denom = freq + index.k1 * (1 - index.b + (index.b * index.docLen[i]) / index.avgdl);
      scores[i] += (idfValue * (freq * (index.k1 + 1))) / denom;
    }
  }
  return scores;
}
