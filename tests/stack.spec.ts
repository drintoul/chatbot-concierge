import { test, expect, request } from "@playwright/test";

const FIRECRAWL = process.env.FIRECRAWL_URL ?? "http://localhost:3002";
const OLLAMA = process.env.OLLAMA_URL ?? "http://localhost:11435";
const CHATBOT_CONCIERGE = process.env.CHATBOT_CONCIERGE_URL ?? "http://localhost:8000";

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

interface RunState {
  status: string;
  steps: { node: string; status: string }[];
  answer?: string | null;
  error?: string | null;
  sources?: string[];
  page_count?: number;
}

async function pollRun(ctx: Awaited<ReturnType<typeof request.newContext>>, id: string): Promise<RunState> {
  for (let i = 0; i < 60; i++) {
    const res = await ctx.get(`${CHATBOT_CONCIERGE}/runs/${id}`);
    expect(res.ok(), `run lookup failed: ${await res.text()}`).toBeTruthy();
    const state = (await res.json()) as RunState;
    if (state.status !== "running") return state;
    await new Promise((r) => setTimeout(r, 2_000));
  }
  throw new Error("run did not finish in 120s");
}

test("chatbot-concierge serves the demo page", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${CHATBOT_CONCIERGE}/`);
  expect(res.ok()).toBeTruthy();
  expect(await res.text()).toContain("chatbot-concierge");
});

const SITE = "https://example.com/";

async function indexSite(ctx: Awaited<ReturnType<typeof request.newContext>>): Promise<RunState> {
  const start = await ctx.post(`${CHATBOT_CONCIERGE}/index`, { data: { url: SITE } });
  expect(start.status()).toBe(202);
  const { run_id } = (await start.json()) as { run_id: string };
  return pollRun(ctx, run_id);
}

test("index workflow crawls and stores a corpus", async () => {
  test.setTimeout(180_000);
  const ctx = await request.newContext();
  const state = await indexSite(ctx);
  expect(state.error).toBeNull();
  expect(state.status).toBe("done");
  expect(state.steps.map((s) => s.node)).toEqual(["start", "crawl", "chunk", "embed", "store"]);

  const sites = (await (await ctx.get(`${CHATBOT_CONCIERGE}/sites`)).json()) as {
    sites: { url: string; chunk_count: number }[];
  };
  expect(sites.sites.some((s) => s.url === SITE)).toBe(true);

  // Homepage screenshot captured during crawl and stored with the corpus.
  const shot = await ctx.get(
    `${CHATBOT_CONCIERGE}/sites/screenshot?url=${encodeURIComponent(SITE)}`
  );
  expect(shot.status()).toBe(200);
  const magic = await shot.body();
  expect([...magic.subarray(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]);
});

test("query workflow answers from the stored corpus", async () => {
  test.setTimeout(240_000);
  const ctx = await request.newContext();
  await indexSite(ctx); // idempotent — rebuilds the corpus

  const start = await ctx.post(`${CHATBOT_CONCIERGE}/query`, {
    data: { url: SITE, question: "What is this domain used for?" },
  });
  expect(start.status()).toBe(202);
  const { run_id } = (await start.json()) as { run_id: string };
  const state = await pollRun(ctx, run_id);

  expect(state.error).toBeNull();
  expect(state.status).toBe("done");
  expect(state.answer ?? "").toMatch(/example/i);
  // answer → verify loop: verify is always last; answer/verify may repeat.
  const nodes = state.steps.map((s) => s.node);
  expect(nodes.slice(0, 3)).toEqual(["start", "load", "retrieve"]);
  expect(nodes[nodes.length - 1]).toBe("verify");
  expect(state.sources?.length ?? 0).toBeGreaterThan(0);

  // Run state persists — checkpointed, inspectable after the fact.
  const again = await ctx.get(`${CHATBOT_CONCIERGE}/runs/${encodeURIComponent(run_id)}`);
  expect((await again.json()).status).toBe("done");
});

// The suite seeds example.com into corpora/events — clean up so test
// runs don't pollute the demo's indexed sites or audit log.
test("delete site removes the test corpus and its audit events", async () => {
  const ctx = await request.newContext();
  const res = await ctx.delete(`${CHATBOT_CONCIERGE}/sites?url=${encodeURIComponent(SITE)}`);
  expect(res.ok(), `delete failed: ${await res.text()}`).toBeTruthy();
  const body = (await res.json()) as { deleted: boolean };
  expect(body.deleted).toBe(true);

  const sites = (await (await ctx.get(`${CHATBOT_CONCIERGE}/sites`)).json()) as {
    sites: { url: string }[];
  };
  expect(sites.sites.some((s) => s.url === SITE)).toBe(false);
});
