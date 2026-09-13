import { test, expect } from "@playwright/test";

// Deliberately not one of the recorded replay questions (see
// content/generated/console.json) -- this must take the live pipeline path
// so it actually exercises retrieval + Groq + Evidence Trail persistence.
// Not asserting the answer is cited: a real run against this small, single-
// filing corpus sometimes honestly answers "not available" for a plain-
// language question even when chunks were retrieved (grounded refusal is
// correct pipeline behavior, not a bug) -- this test verifies the console-
// to-evidence-trail plumbing, not retrieval recall. Kept narrow deliberately:
// a broad question (e.g. "main risk factors") can trigger multiple reflect
// loops and aggregate enough context to hit Groq's 8000 TPM on-demand
// ceiling (see lib/rag/agent.ts's friendlyErrorMessage).
const LIVE_QUESTION = "What does NVIDIA say about its data center segment?";

test("console query streams a live answer and links to a working evidence trail", async ({
  page,
}) => {
  await page.goto("/console");

  const input = page.getByPlaceholder("Ask any question about the indexed filing");
  await input.fill(LIVE_QUESTION);
  await page.getByRole("button", { name: "Execute" }).click();

  const evidenceLink = page.getByRole("link", { name: /view evidence trail/i });
  await expect(evidenceLink).toBeVisible({ timeout: 100_000 });

  const retrievedCount = await page
    .locator("span", { hasText: "Retrieved" })
    .locator("b")
    .innerText();
  expect(Number(retrievedCount)).toBeGreaterThan(0);

  await evidenceLink.click();
  await expect(page).toHaveURL(/\/evidence\/.+/);

  await expect(page.getByText("Plan", { exact: true })).toBeVisible();
  await expect(page.getByText(/Retrieved chunks/i)).toBeVisible();
  // The chunk count on the evidence page should match what the console
  // showed for this same run -- confirms the persisted record is the real
  // one, not a stale/mismatched read.
  await expect(page.getByText(`${retrievedCount} retrieved`)).toBeVisible();
});
