#!/usr/bin/env node
/**
 * Enforces the Measured/Illustrative provenance discipline (deployment_roadmap.md
 * §4/§7) at the tooling level: every content/generated/*.json artifact must carry
 * a provenance tag, and no previously-removed fabricated constant can silently
 * come back. Scoped to the three artifacts precompute.py actually produces with
 * known shapes -- not a generic schema validator, see §9 Phase 5.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

const GENERATED_DIR = join(import.meta.dirname, "..", "content", "generated");

// Removed for being explicitly mocked with no stored historical run (§9 Phase 1).
const BANNED_KEYS = ["CACHE_TOKEN_SAVINGS", "LATENCY_BREAKDOWN"];

const failures = [];

function load(file) {
  return JSON.parse(readFileSync(join(GENERATED_DIR, file), "utf-8"));
}

function requireProvenance(path, obj) {
  if (typeof obj !== "object" || obj === null) {
    failures.push(`${path}: expected an object, got ${typeof obj}`);
    return;
  }
  if (!("provenance" in obj)) {
    failures.push(`${path}: missing "provenance" field`);
  } else if (!["measured", "illustrative", "live"].includes(obj.provenance)) {
    failures.push(`${path}: unexpected provenance value "${obj.provenance}"`);
  }
}

// corpus.json tags provenance per top-level section, not once globally.
const corpus = load("corpus.json");
requireProvenance("corpus.json:filings", corpus.filings);
requireProvenance("corpus.json:supported_filing_types", corpus.supported_filing_types);

// evaluation.json and system.json tag provenance once at the top.
requireProvenance("evaluation.json", load("evaluation.json"));
requireProvenance("system.json", load("system.json"));

const allText = ["corpus.json", "evaluation.json", "system.json", "console.json"]
  .map((f) => readFileSync(join(GENERATED_DIR, f), "utf-8"))
  .join("\n");
for (const key of BANNED_KEYS) {
  if (allText.includes(key)) {
    failures.push(`Banned mock constant "${key}" reappeared in a generated artifact`);
  }
}

if (failures.length > 0) {
  console.error("Data integrity check FAILED:\n" + failures.map((f) => `  - ${f}`).join("\n"));
  process.exit(1);
}
console.log("Data integrity check passed: provenance present, no banned mock constants.");
