import "server-only";
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

// Xenova/ms-marco-MiniLM-L-6-v2 is the ONNX conversion of
// cross-encoder/ms-marco-MiniLM-L-6-v2, the reranker the Python backend
// uses. It is a single-output regression head, so the raw logit (not a
// softmaxed pipeline() score) is the ranking signal, matching
// CrossEncoder.predict() on the Python side.
const MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2";

let modelPromise: Promise<{ model: PreTrainedModel; tokenizer: PreTrainedTokenizer }> | null = null;

function getReranker() {
  if (!modelPromise) {
    modelPromise = Promise.all([
      AutoModelForSequenceClassification.from_pretrained(MODEL_ID),
      AutoTokenizer.from_pretrained(MODEL_ID),
    ]).then(([model, tokenizer]) => ({ model, tokenizer }));
  }
  return modelPromise;
}

export async function rerankScores(query: string, documents: string[]): Promise<number[]> {
  if (documents.length === 0) return [];
  const { model, tokenizer } = await getReranker();
  const scores: number[] = [];
  for (const doc of documents) {
    const inputs = tokenizer(query, { text_pair: doc, padding: true, truncation: true });
    const output = await model(inputs);
    scores.push(output.logits.data[0] as number);
  }
  return scores;
}
