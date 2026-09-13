import "server-only";
import fs from "node:fs";
import path from "node:path";
import type { SystemData, EvaluationData, CorpusData } from "./types";

const CONTENT_DIR = path.join(process.cwd(), "content", "generated");

function readJson<T>(filename: string): T {
  const filePath = path.join(CONTENT_DIR, filename);
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

/**
 * Reads build-time JSON artifacts produced by `uv run scripts/precompute.py`.
 * Server-only, synchronous, no runtime fetch — see deployment_roadmap.md §7.
 */
export function getSystemData(): SystemData {
  return readJson<SystemData>("system.json");
}

export function getEvaluationData(): EvaluationData {
  return readJson<EvaluationData>("evaluation.json");
}

export function getCorpusData(): CorpusData {
  return readJson<CorpusData>("corpus.json");
}
