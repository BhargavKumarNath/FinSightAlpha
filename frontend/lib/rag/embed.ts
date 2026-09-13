import "server-only";
import "./onnxEnv";
import { pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";

// Xenova/all-MiniLM-L6-v2 is the ONNX conversion of the same
// all-MiniLM-L6-v2 model the Python backend embeds with
// (src/retrieval/hybrid_retriever.py), so query vectors here land in the
// same space as the corpus vectors precompute.py exported.
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

let embedderPromise: Promise<FeatureExtractionPipeline> | null = null;

function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline("feature-extraction", MODEL_ID);
  }
  return embedderPromise;
}

export async function embedQuery(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}
