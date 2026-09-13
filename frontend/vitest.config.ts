import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // e2e/ holds Playwright specs, run via `npx playwright test`, not vitest.
    include: ["lib/**/*.test.ts"],
  },
});
