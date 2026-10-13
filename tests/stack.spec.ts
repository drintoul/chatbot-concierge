import { test, expect, request } from "@playwright/test";

const FIRECRAWL = process.env.FIRECRAWL_URL ?? "http://localhost:3002";
const OLLAMA = process.env.OLLAMA_URL ?? "http://localhost:11435";
const AGENT = process.env.AGENT_URL ?? "http://localhost:8000";

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
}

async function pollRun(ctx: Awaited<ReturnType<typeof request.newContext>>, id: string): Promise<RunState> {
  for (let i = 0; i < 60; i++) {
    const res = await ctx.get(`${AGENT}/runs/${id}`);
    expect(res.ok(), `run lookup failed: ${await res.text()}`).toBeTruthy();
    const state = (await res.json()) as RunState;
    if (state.status !== "running") return state;
    await new Promise((r) => setTimeout(r, 2_000));
  }
  throw new Error("run did not finish in 120s");
}

test("agent runs deterministic scrape→answer pipeline", async () => {
  test.setTimeout(180_000);
  const ctx = await request.newContext();
  const start = await ctx.post(`${AGENT}/runs`, {
    data: { url: "https://example.com", question: "What is this domain used for?" },
  });
  expect(start.status()).toBe(202);
  const { run_id } = (await start.json()) as { run_id: string };

  const state = await pollRun(ctx, run_id);
  expect(state.error).toBeNull();
  expect(state.status).toBe("done");
  expect(state.answer ?? "").toMatch(/example/i);
  // Fixed graph order — determinism means the node trace is identical every run.
  expect(state.steps.map((s) => s.node)).toEqual(["start", "scrape", "chunk", "embed", "retrieve", "answer"]);
});

test("agent checkpoints run state for later inspection", async () => {
  test.setTimeout(180_000);
  const ctx = await request.newContext();
  const start = await ctx.post(`${AGENT}/runs`, {
    data: { url: "https://example.com" }, // no question → retrieve skips, no answer node
  });
  const { run_id } = (await start.json()) as { run_id: string };
  const state = await pollRun(ctx, run_id);

  expect(state.status).toBe("done");
  expect(state.steps.map((s) => s.node)).toEqual(["start", "scrape", "chunk", "embed", "retrieve"]);
  expect(state.steps.at(-1)?.status).toBe("skipped");

  // Same run, re-fetched — state comes from the SQLite checkpointer.
  const again = await ctx.get(`${AGENT}/runs/${run_id}`);
  expect((await again.json()).status).toBe("done");

  const list = await ctx.get(`${AGENT}/runs`);
  const runs = ((await list.json()) as { runs: RunState[] }).runs;
  expect(runs.length).toBeGreaterThan(0);
});
