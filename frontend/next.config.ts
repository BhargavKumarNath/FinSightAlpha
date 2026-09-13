import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Explicitly bundle files the chat route's serverless function needs but
  // that Next's tracer can't discover on its own:
  //  - vector_index.json is read via fs.readFileSync + path.join at request
  //    time (lib/rag/vectorIndex.ts), not imported as a module.
  //  - onnxruntime-node's .node addon dlopen()s libonnxruntime.so.1 as a
  //    plain OS-level dynamic link at runtime, invisible to JS-level tracing
  //    (confirmed missing on a real deploy: "libonnxruntime.so.1: cannot
  //    open shared object file").
  outputFileTracingIncludes: {
    "app/api/chat/route": [
      "./content/generated/vector_index.json",
      "./node_modules/onnxruntime-node/bin/napi-v6/linux/x64/*",
    ],
  },
};

export default nextConfig;
