import "server-only";
import fs from "node:fs";
import path from "node:path";
import type { VectorIndexFile } from "./types";

let cached: VectorIndexFile | null = null;

export function getVectorIndex(): VectorIndexFile {
  if (cached) return cached;
  const filePath = path.join(process.cwd(), "content", "generated", "vector_index.json");
  cached = JSON.parse(fs.readFileSync(filePath, "utf-8")) as VectorIndexFile;
  return cached;
}
