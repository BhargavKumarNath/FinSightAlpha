import { describe, expect, it } from "vitest";
import { buildBm25Index, bm25Scores, tokenize } from "./bm25";

/**
 * Ground truth from Python's rank_bm25 (the library src/retrieval/
 * hybrid_retriever.py actually uses), computed via:
 *   BM25Okapi([d.lower().split() for d in docs]).get_scores(query.lower().split())
 * on this exact corpus/query — see deployment_roadmap.md §10. Confirms the
 * TS port reproduces the same scores, not just a plausible approximation.
 */
const docs = [
  "revenue grew significantly in the fiscal year driven by data center demand",
  "net income increased due to strong gross margins across all segments",
  "the company invests heavily in research and development for new chips",
  "supply chain constraints affected gaming segment revenue in the quarter",
  "data center revenue reached a record high this fiscal year",
];
const EXPECTED_SCORES = [1.50358, 0.0, 0.0, 0.2409, 1.633199];

describe("bm25Scores (ported from rank_bm25.BM25Okapi)", () => {
  it("matches rank_bm25's scores to 3 decimal places", () => {
    const index = buildBm25Index(docs);
    const scores = bm25Scores(index, tokenize("data center revenue fiscal year"));
    scores.forEach((s, i) => expect(s).toBeCloseTo(EXPECTED_SCORES[i], 3));
  });

  it("matches rank_bm25's top-ranked ordering", () => {
    const index = buildBm25Index(docs);
    const scores = bm25Scores(index, tokenize("data center revenue fiscal year"));
    const ranked = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
    const expectedRanked = EXPECTED_SCORES.map((s, i) => i).sort(
      (a, b) => EXPECTED_SCORES[b] - EXPECTED_SCORES[a],
    );
    expect(ranked).toEqual(expectedRanked);
  });

  it("scores a doc with zero query-term overlap as 0", () => {
    const index = buildBm25Index(docs);
    const scores = bm25Scores(index, tokenize("data center revenue fiscal year"));
    expect(scores[1]).toBe(0);
    expect(scores[2]).toBe(0);
  });
});

describe("tokenize", () => {
  it("lowercases and splits on whitespace, matching Python's .lower().split()", () => {
    expect(tokenize("Data  Center\tRevenue")).toEqual(["data", "center", "revenue"]);
  });
});
