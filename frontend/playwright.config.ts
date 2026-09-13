import { defineConfig, devices } from "@playwright/test";

/**
 * §10's "one critical user path": Console -> live query -> streamed answer
 * -> Evidence Trail. Runs against the real local pipeline (ONNX embed/rerank,
 * bundled vector index, real Groq calls) -- there is no staging environment
 * to point this at (see deployment_roadmap.md §9, Phase 5 deviations), so it
 * doubles as the closest thing to the §10 "integration" test too.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  fullyParallel: false,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // Production build, not `next dev`: dev's on-demand compilation and
    // React StrictMode double-invoked effects were adding tens of seconds
    // of jitter to a component that mounts by auto-running a replay query.
    command: "npm run build && npm run start",
    url: "http://127.0.0.1:3000",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
