import { test, expect, request } from "@playwright/test";

const FIRECRAWL = process.env.FIRECRAWL_URL ?? "http://localhost:3002";
const OLLAMA = process.env.OLLAMA_URL ?? "http://localhost:11435";

test("ollama answers /api/tags", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${OLLAMA}/api/tags`);
  expect(res.ok()).toBeTruthy();
  const body = (await res.json()) as { models: unknown[] };
  expect(Array.isArray(body.models)).toBe(true);
});

test("firecrawl api is up", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${FIRECRAWL}/`);
  // Root route may 404 — anything under 500 proves the harness is serving.
  expect(res.status()).toBeLessThan(500);
});

test("firecrawl scrapes a public page to markdown", async () => {
  test.setTimeout(120_000);
  const ctx = await request.newContext();
  const res = await ctx.post(`${FIRECRAWL}/v2/scrape`, {
    data: { url: "https://example.com", formats: ["markdown"] },
  });
  expect(res.ok(), `scrape failed: ${await res.text()}`).toBeTruthy();
  const body = (await res.json()) as {
    success: boolean;
    data: { markdown?: string; metadata?: { title?: string } };
  };
  expect(body.success).toBe(true);
  expect(body.data.markdown ?? "").toContain("documentation examples");
  expect(body.data.metadata?.title ?? "").toContain("Example Domain");
});
