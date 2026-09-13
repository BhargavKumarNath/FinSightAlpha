"use client";

import type { EvidenceRecord } from "@/lib/data/types";

/**
 * No server-side persistence exists for this build (no Vercel KV / DB
 * account provisioned, see deployment_roadmap.md §9, Phase 3). Evidence
 * Trail records live in this browser's localStorage only, keyed by a
 * generated query id: a page reload or the same query on another device
 * won't find them. Documented limitation, not a bug.
 */

const RECORD_PREFIX = "finsight:evidence:";
const LATEST_KEY = "finsight:evidence:latest";
const UPDATED_EVENT = "finsight:evidence-updated";
const MAX_RECORDS = 30;
const INDEX_KEY = "finsight:evidence:index";

function isBrowser() {
  return typeof window !== "undefined";
}

function readIndex(): string[] {
  if (!isBrowser()) return [];
  try {
    const raw = window.localStorage.getItem(INDEX_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function writeIndex(ids: string[]) {
  if (!isBrowser()) return;
  window.localStorage.setItem(INDEX_KEY, JSON.stringify(ids));
}

export function saveEvidence(record: EvidenceRecord) {
  if (!isBrowser()) return;
  try {
    window.localStorage.setItem(RECORD_PREFIX + record.queryId, JSON.stringify(record));
    window.localStorage.setItem(LATEST_KEY, record.queryId);

    const ids = readIndex().filter((id) => id !== record.queryId);
    ids.unshift(record.queryId);
    while (ids.length > MAX_RECORDS) {
      const evicted = ids.pop();
      if (evicted) window.localStorage.removeItem(RECORD_PREFIX + evicted);
    }
    writeIndex(ids);

    window.dispatchEvent(new Event(UPDATED_EVENT));
  } catch {
    // localStorage unavailable (private mode, quota, etc.) -- Evidence
    // Trail simply won't have this run available. Non-fatal.
  }
}

export function getEvidence(queryId: string): EvidenceRecord | null {
  if (!isBrowser()) return null;
  try {
    const raw = window.localStorage.getItem(RECORD_PREFIX + queryId);
    return raw ? (JSON.parse(raw) as EvidenceRecord) : null;
  } catch {
    return null;
  }
}

export function getLatestEvidenceId(): string | null {
  if (!isBrowser()) return null;
  try {
    return window.localStorage.getItem(LATEST_KEY);
  } catch {
    return null;
  }
}

export function onEvidenceUpdated(callback: () => void): () => void {
  if (!isBrowser()) return () => {};
  window.addEventListener(UPDATED_EVENT, callback);
  return () => window.removeEventListener(UPDATED_EVENT, callback);
}
