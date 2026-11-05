import { test, expect, request } from "@playwright/test";

const OLLAMA = process.env.OLLAMA_URL ?? "http://localhost:11434";
const CHATBOT_CONCIERGE = process.env.CHATBOT_CONCIERGE_URL ?? "http://localhost:8000";

test("ollama answers /api/tags", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${OLLAMA}/api/tags`);
  expect(res.ok()).toBeTruthy();
  const body = (await res.json()) as { models: unknown[] };
  expect(Array.isArray(body.models)).toBe(true);
});

interface RunState {
  status: string;
  steps: { node: string; status: string }[];
  answer?: string | null;
  error?: string | null;
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

test("chatbot-concierge serves the public chat page", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${CHATBOT_CONCIERGE}/`);
  expect(res.ok()).toBeTruthy();
  expect(await res.text()).toContain("concierge");
});

test("chatbot-concierge lists corpora", async () => {
  const ctx = await request.newContext();
  const res = await ctx.get(`${CHATBOT_CONCIERGE}/sites`);
  expect(res.ok()).toBeTruthy();
  const body = (await res.json()) as { sites: unknown[] };
  expect(Array.isArray(body.sites)).toBe(true);
});

// A URL that will never be indexed — corpora come from the extractor
// service, so querying one must fail cleanly inside the run, not 500/hang.
const SITE = "https://unindexed.invalid/";
test("query on an unindexed site fails cleanly", async () => {
  const ctx = await request.newContext();
  const start = await ctx.post(`${CHATBOT_CONCIERGE}/query`, {
    data: { url: SITE, question: "What is this domain used for?", token: "BADTOKEN" },
  });
  expect(start.status()).toBe(202);
  const { run_id } = (await start.json()) as { run_id: string };
  const state = await pollRun(ctx, run_id);
  expect(state.status).toBe("failed");
  expect(state.error ?? "").toMatch(/not indexed/i);
  expect(state.steps.map((s) => s.node)).toContain("load");
});

test("delete on an unknown site reports nothing deleted", async () => {
  const ctx = await request.newContext();
  const res = await ctx.delete(
    `${CHATBOT_CONCIERGE}/sites?url=${encodeURIComponent(SITE)}`
  );
  expect(res.ok(), `delete failed: ${await res.text()}`).toBeTruthy();
  const body = (await res.json()) as { deleted: boolean };
  expect(body.deleted).toBe(false);

  const audit = await ctx.delete(
    `${CHATBOT_CONCIERGE}/audit?url=${encodeURIComponent(SITE)}`
  );
  expect(audit.ok()).toBeTruthy();
});
