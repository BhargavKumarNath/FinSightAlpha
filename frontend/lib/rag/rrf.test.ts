import { describe, expect, it } from "vitest";
import { reciprocalRankFusion } from "./rrf";

describe("reciprocalRankFusion", () => {
  it("ranks a doc that leads both lists first", () => {
    const dense = ["a", "b", "c"];
    const sparse = ["a", "b", "c"];
    const fused = reciprocalRankFusion(dense, sparse, 3);
    expect(fused.map((f) => f.id)).toEqual(["a", "b", "c"]);
  });

  it("ties two docs that swap ranks across lists", () => {
    // b: dense rank 1, sparse rank 2. c: dense rank 2, sparse rank 1.
    // 1/(k+1+1) + 1/(k+2+1) == 1/(k+2+1) + 1/(k+1+1) -> exact tie.
    const dense = ["a", "b", "c"];
    const sparse = ["a", "c", "b"];
    const fused = reciprocalRankFusion(dense, sparse, 3);
    const b = fused.find((f) => f.id === "b")!;
    const c = fused.find((f) => f.id === "c")!;
    expect(b.rrfScore).toBeCloseTo(c.rrfScore, 10);
  });

  it("gives a doc found by both retrievers a higher score than one found by only one", () => {
    const dense = ["a", "b"];
    const sparse = ["a", "z"];
    const fused = reciprocalRankFusion(dense, sparse, 10);
    const a = fused.find((f) => f.id === "a")!;
    const b = fused.find((f) => f.id === "b")!;
    const z = fused.find((f) => f.id === "z")!;
    expect(a.rrfScore).toBeGreaterThan(b.rrfScore);
    expect(a.rrfScore).toBeGreaterThan(z.rrfScore);
    // rank 0 in dense == rank 1 in sparse (b vs z), so b and z tie.
    expect(b.rrfScore).toBeCloseTo(z.rrfScore, 10);
  });

  it("respects k=60 exactly (matches hybrid_retriever.py's RRF_K)", () => {
    const fused = reciprocalRankFusion(["a"], [], 1);
    // rank(a) = 0 in dense, absent from sparse -> score = 1 / (60 + 0 + 1)
    expect(fused[0].rrfScore).toBeCloseTo(1 / 61, 10);
  });

  it("truncates to topN", () => {
    const dense = ["a", "b", "c", "d"];
    const fused = reciprocalRankFusion(dense, [], 2);
    expect(fused).toHaveLength(2);
    expect(fused.map((f) => f.id)).toEqual(["a", "b"]);
  });
});
