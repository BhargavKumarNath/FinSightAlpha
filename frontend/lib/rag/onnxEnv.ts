import "server-only";
import { env } from "@huggingface/transformers";

/**
 * transformers.js defaults its model cache to a path relative to its own
 * install location, which is read-only in a deployed serverless function
 * (confirmed on Vercel: "ENOENT: no such file or directory, mkdir
 * '/var/task/node_modules/@huggingface/transformers/.cache'"). /tmp is the
 * only writable path in that environment. Side-effecting import: load this
 * before creating any pipeline (embed.ts, rerank.ts).
 */
env.cacheDir = "/tmp/.transformers-cache";
