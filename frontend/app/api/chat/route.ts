import { runAgent } from "@/lib/rag/agent";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const encoder = new TextEncoder();

export async function POST(request: Request) {
  let query: string;
  try {
    const body = (await request.json()) as { query?: string };
    query = (body.query ?? "").trim();
  } catch {
    return new Response("Invalid JSON body.", { status: 400 });
  }

  if (!query) {
    return new Response("Missing query.", { status: 400 });
  }
  if (query.length > 500) {
    return new Response("Query too long.", { status: 400 });
  }

  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const event of runAgent(query)) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown pipeline error.";
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "error", message })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
