import "server-only";
import Groq from "groq-sdk";

// Model split and prompts ported from src/agents/langgraph_agent.py and
// src/optimization/config.py: a fast, small model for planning/routing
// tasks, a larger model reserved for the actual synthesis step. The
// Python config's model names (llama-3.3-70b-versatile,
// llama-3.1-8b-instant) have since been retired from Groq's catalog;
// these are their closest current equivalents by role and size.
const MODEL_HEAVY = "openai/gpt-oss-120b";
const MODEL_LIGHT = "openai/gpt-oss-20b";
const TEMPERATURE = 0;

let client: Groq | null = null;

function getClient(): Groq {
  if (!client) {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) throw new Error("GROQ_API_KEY is not set.");
    client = new Groq({ apiKey });
  }
  return client;
}

async function complete(model: string, systemPrompt: string, userContent?: string, jsonMode = false): Promise<string> {
  const messages: Groq.Chat.ChatCompletionMessageParam[] = [{ role: "system", content: systemPrompt }];
  if (userContent) messages.push({ role: "user", content: userContent });

  const completion = await getClient().chat.completions.create({
    model,
    messages,
    temperature: TEMPERATURE,
    response_format: jsonMode ? { type: "json_object" } : undefined,
    // gpt-oss models have a built-in browser_search tool baked into their
    // chat template and will sometimes emit a tool call for it even with
    // no tools configured, which Groq then rejects outright. Low
    // reasoning effort reliably avoids that path for this task shape.
    reasoning_effort: "low",
  });
  return completion.choices[0]?.message?.content ?? "";
}

const PLANNER_PROMPT =
  "You are a Senior Financial Strategist. Analyze the user's query and formulate a 1-3 step " +
  "retrieval plan to answer it completely. Output ONLY the numbered list. No filler.";

const REWRITER_PROMPT_TEMPLATE = (plan: string, reflection: string) =>
  "You are a skilled Query Decomposition Agent. The user wants to answer a complex, multi-hop question. " +
  "Your job is to break the main query into separate atomic search queries optimized for a semantic vector database. " +
  "CRITICAL RULES:\n" +
  "1. SEQUENTIAL DEPENDENCIES: If identifying an entity is required before answering a subsequent question, branch the questions sequentially! (e.g. ['Company that acquired Figma in 2022', 'CEO of Adobe', 'Open-source framework created by Adobe']).\n" +
  "2. Generate NATURAL LANGUAGE queries only. NEVER generate SQL, code, or boolean expressions. Expand abbreviations automatically.\n" +
  "3. Focus on entities, years, financial metrics, and specific risk factors.\n" +
  `Previous plan: ${plan}\n` +
  `Feedback from reflection (if any): ${reflection}\n\n` +
  'Return JSON exact schema: {"queries": ["query 1", "query 2"]}';

const REASONER_PROMPT_TEMPLATE = (query: string, context: string) =>
  "You are an elite Financial Analyst. Answer the user's query using ONLY the provided retrieved chunks. " +
  "CRITICAL RULES:\n" +
  "1. You MUST cite the exact [Doc X: source_file] for every claim you make.\n" +
  "2. Do not mix sources without citing both.\n" +
  "3. If specific data (dollar amounts, percentages, dates) appears in the chunks, ALWAYS include it in your answer.\n" +
  "4. If the documents genuinely do not contain the answer, explicitly state so, but first carefully re-read ALL chunks.\n" +
  "5. Do not fabricate information.\n" +
  "6. Provide specific numbers and details whenever they appear in the context.\n" +
  `\nUser Query: ${query}\n\n` +
  `--- Retrieved Context ---\n${context}`;

const REFLECTOR_PROMPT_TEMPLATE = (context: string, draft: string) =>
  "You are a strict Hallucination Checker & Evaluator. You are reviewing a draft answer against the provided context. " +
  "1. Is the answer directly addressing the user's query based ONLY on the context?\n" +
  "2. Are there any fabricated facts or numbers not present in the context?\n" +
  "3. Does the draft have missing information that requires another search?\n" +
  `\nContext:\n${context}\n\nDraft Answer:\n${draft}\n\n` +
  "Output JSON strictly with keys: 'is_grounded' (boolean), 'needs_more_info' (boolean), 'feedback' (string).";

export async function planQuery(query: string): Promise<string> {
  return complete(MODEL_LIGHT, PLANNER_PROMPT, query);
}

export async function rewriteQuery(query: string, plan: string, reflection: string): Promise<string[]> {
  const prompt = REWRITER_PROMPT_TEMPLATE(plan, reflection);
  const raw = await complete(MODEL_LIGHT, prompt, `Original Query: ${query}`, true);
  try {
    const parsed = JSON.parse(raw) as { queries?: string[] };
    return parsed.queries && parsed.queries.length > 0 ? parsed.queries : [query];
  } catch {
    return [query];
  }
}

// gpt-oss models sometimes render citation brackets and hyphens as
// fullwidth/CJK punctuation (e.g. "【Doc 1: full‑submission.txt】") even when the
// prompt spells out plain ASCII "[Doc N: source]". Both countCitations()
// server-side and CitedResponse client-side match literal "[" / "]", so
// normalize back to ASCII right at the source.
function normalizeCitationPunctuation(text: string): string {
  return text
    .replace(/[【［]/g, "[")
    .replace(/[】］]/g, "]")
    .replace(/[‐‑‒–—]/g, "-");
}

export async function reason(query: string, context: string): Promise<string> {
  const prompt = REASONER_PROMPT_TEMPLATE(query, context);
  const draft = await complete(MODEL_HEAVY, prompt);
  return normalizeCitationPunctuation(draft);
}

export type Reflection = { isGrounded: boolean; needsMoreInfo: boolean; feedback: string };

export async function reflect(context: string, draft: string): Promise<Reflection> {
  const prompt = REFLECTOR_PROMPT_TEMPLATE(context, draft);
  const raw = await complete(MODEL_LIGHT, prompt, undefined, true);
  try {
    const parsed = JSON.parse(raw) as { is_grounded?: boolean; needs_more_info?: boolean; feedback?: string };
    return {
      isGrounded: parsed.is_grounded ?? true,
      needsMoreInfo: parsed.needs_more_info ?? false,
      feedback: parsed.feedback ?? "",
    };
  } catch {
    return { isGrounded: true, needsMoreInfo: false, feedback: "Parse failed." };
  }
}
