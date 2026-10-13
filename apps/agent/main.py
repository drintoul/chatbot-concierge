"""Deterministic LangGraph agent service.

Pipeline: scrape (firecrawl) -> chunk -> embed (ollama) -> retrieve -> answer.
Every edge is fixed code — the LLM never decides control flow. The whole run
is checkpointed to SQLite (langgraph-checkpoint-sqlite), so GET /runs/{id}
returns the persisted state at any point: each node appends to `steps`,
giving you stateful, inspectable progression through the process.

Determinism: temperature=0 LLM calls, deterministic chunker, pure-math
cosine ranking, fixed prompts. Same inputs -> same outputs.
"""

import asyncio
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, TypedDict

import aiosqlite
import httpx
import operator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, HttpUrl

OLLAMA = os.environ["OLLAMA_BASE_URL"].rstrip("/")
FIRECRAWL = os.environ["FIRECRAWL_URL"].rstrip("/")
CHAT_MODEL = os.environ["OLLAMA_CHAT_MODEL"]
EMBED_MODEL = os.environ["OLLAMA_EMBED_MODEL"]
DB_PATH = os.environ.get("AGENT_DB_PATH", "/data/langgraph.db")
TOP_K = 4
MIN_SIMILARITY = 0.15

http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
_graph = None
_saver = None
_pending: set[str] = set()  # run_ids accepted but not yet checkpointed


class AgentState(TypedDict, total=False):
    url: str
    question: str | None
    markdown: str
    title: str | None
    chunks: list[str]
    embeddings: list[list[float]]
    ranked: list[dict[str, Any]]
    answer: str | None
    # operator.add reducer: every node appends a trace entry → visible
    # progression of the run inside the checkpoint.
    steps: Annotated[list[dict[str, Any]], operator.add]
    status: Literal["running", "done", "failed"]
    error: str | None


def step(node: str, status: str, **detail: Any) -> dict[str, Any]:
    return {"node": node, "status": status, "at": time.time(), **detail}


def fail(state: AgentState, node: str, err: Exception | str) -> AgentState:
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    return {"status": "failed", "error": msg, "steps": [step(node, "failed", error=msg[:300])]}


def bail(state: AgentState) -> bool:
    return state.get("status") == "failed"


# ----------------------------------------------------------------- nodes

async def scrape(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    t0 = time.time()
    try:
        res = await http.post(
            f"{FIRECRAWL}/v2/scrape",
            json={"url": state["url"], "formats": ["markdown"], "onlyMainContent": True},
        )
        res.raise_for_status()
        body = res.json()
        if not body.get("success"):
            raise RuntimeError(f"firecrawl scrape failed: {str(body)[:200]}")
        data = body["data"]
        md = data.get("markdown") or ""
        if len(md) < 40:
            raise RuntimeError("scraped page has too little content")
        return {
            "markdown": md,
            "title": (data.get("metadata") or {}).get("title"),
            "steps": [step("scrape", "ok", ms=round((time.time() - t0) * 1000), chars=len(md))],
        }
    except Exception as e:
        return fail(state, "scrape", e)


import re

_IMAGE_LINE = re.compile(r"^\s*!\[.*\]\(.*?\)\s*\\?\s*$")
_LINK_LINE = re.compile(r"^\s*\[.*\]\(.*?\)\s*\\?\s*$")
_BOILERPLATE = re.compile(
    r"consent management|cookies?|all rights reserved|powered by|newsletter|"
    r"privacy policy|terms of (service|use)|accept all|skip to (main )?(content|nav)",
    re.I,
)


def _clean(para: str) -> str:
    """Drop image-only/link-only lines; returns '' for boilerplate/nav."""
    lines = [
        ln
        for ln in para.splitlines()
        if not _IMAGE_LINE.match(ln) and not _LINK_LINE.match(ln)
    ]
    text = "\n".join(lines).strip()
    # Short paragraphs stay — "$275 per night +" and "Sundays - Thursdays
    # only" are exactly the qualifiers that must survive chunking.
    if not text or _BOILERPLATE.search(text):
        return ""
    return text


def _chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Deterministic paragraph-aware chunker with boilerplate stripping."""
    paras = [c for p in text.split("\n\n") if (c := _clean(p))]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            cur = f"{tail}\n\n{p}" if tail else p
            while len(cur) > size:
                chunks.append(cur[:size])
                cur = cur[size - overlap :]
    if cur:
        chunks.append(cur)
    return chunks


async def chunk(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    chunks = _chunk(state["markdown"])
    return {
        "chunks": chunks,
        "steps": [step("chunk", "ok", count=len(chunks))],
    }


async def embed(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    t0 = time.time()
    try:
        res = await http.post(
            f"{OLLAMA}/v1/embeddings",
            json={"model": EMBED_MODEL, "input": state["chunks"]},
        )
        res.raise_for_status()
        embs = [d["embedding"] for d in res.json()["data"]]
        return {
            "embeddings": embs,
            "steps": [step("embed", "ok", ms=round((time.time() - t0) * 1000), count=len(embs), model=EMBED_MODEL)],
        }
    except Exception as e:
        return fail(state, "embed", e)


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def retrieve(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    q = state.get("question")
    if not q:
        return {"steps": [step("retrieve", "skipped", reason="no_question")]}
    try:
        res = await http.post(
            f"{OLLAMA}/v1/embeddings",
            json={"model": EMBED_MODEL, "input": [q]},
        )
        res.raise_for_status()
        qv = res.json()["data"][0]["embedding"]
        scored = sorted(
            (
                {"index": i, "score": _cos(qv, e), "text": c}
                for i, (c, e) in enumerate(zip(state["chunks"], state["embeddings"]))
            ),
            key=lambda r: (-r["score"], r["index"]),
        )
        ranked = [r for r in scored if r["score"] >= MIN_SIMILARITY][:TOP_K]
        return {
            "ranked": ranked,
            "steps": [step("retrieve", "ok", top_k=len(ranked), best=round(scored[0]["score"], 4) if scored else 0)],
        }
    except Exception as e:
        return fail(state, "retrieve", e)


SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied scraped-page context. "
    "If the context contains relevant information, summarize it in 1-3 "
    "sentences, sticking strictly to what the context states. Preserve "
    "pricing and availability qualifiers exactly as written (per person, "
    "per night, from/+ pricing, taxes, gratuities, minimums, day or date "
    "restrictions). If the context contains nothing relevant, say exactly: "
    "\"I don't have that information.\" Never invent facts, dates, prices, "
    "or availability."
)


async def answer(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    t0 = time.time()
    ranked = state.get("ranked") or []
    if not ranked:
        return {
            "answer": "I don't have that information.",
            "status": "done",
            "steps": [step("answer", "ok", grounded=False)],
        }
    context = "\n\n---\n\n".join(r["text"] for r in ranked)
    try:
        res = await http.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": CHAT_MODEL,
                "temperature": 0,  # deterministic decode
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}"},
                ],
            },
        )
        res.raise_for_status()
        text = res.json()["choices"][0]["message"]["content"].strip()
        return {
            "answer": text,
            "status": "done",
            "steps": [step("answer", "ok", ms=round((time.time() - t0) * 1000), model=CHAT_MODEL)],
        }
    except Exception as e:
        return fail(state, "answer", e)


def _route(state: AgentState) -> str:
    if bail(state):
        return END
    return "answer" if state.get("question") else END


def _finish(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    return {"status": "done"}


# ----------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _saver
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = await aiosqlite.connect(DB_PATH)
    saver = AsyncSqliteSaver(conn)
    _saver = saver
    builder = StateGraph(AgentState)
    for name, fn in [("scrape", scrape), ("chunk", chunk), ("embed", embed), ("retrieve", retrieve), ("answer", answer), ("finish", _finish)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "scrape")
    builder.add_edge("scrape", "chunk")
    builder.add_edge("chunk", "embed")
    builder.add_edge("embed", "retrieve")
    builder.add_conditional_edges("retrieve", _route, {"answer": "answer", END: "finish"})
    builder.add_edge("answer", "finish")
    builder.add_edge("finish", END)
    _graph = builder.compile(checkpointer=saver)
    yield
    await http.aclose()


app = FastAPI(title="hustle-agent", lifespan=lifespan)


class RunRequest(BaseModel):
    url: HttpUrl
    question: str | None = None


async def _run(run_id: str, url: str, question: str | None) -> None:
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        await _graph.ainvoke(
            {"url": url, "question": question, "status": "running", "steps": [step("start", "ok")]},
            cfg,
        )
    except Exception as e:
        # Node-level errors are caught inside nodes; this guards harness faults.
        try:
            await _graph.aupdate_state(cfg, {"status": "failed", "error": str(e)[:300]})
        except Exception:
            pass
    finally:
        _pending.discard(run_id)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/runs", status_code=202)
async def start_run(req: RunRequest) -> dict[str, str]:
    run_id = uuid.uuid4().hex
    _pending.add(run_id)
    asyncio.create_task(_run(run_id, str(req.url), req.question))
    return {"run_id": run_id, "status": "running"}


def _snapshot(run_id: str, v: dict[str, Any]) -> dict[str, Any]:
    v = v or {}
    return {
        "run_id": run_id,
        "status": v.get("status", "running"),
        "url": v.get("url"),
        "question": v.get("question"),
        "title": v.get("title"),
        "steps": v.get("steps", []),
        "chunk_count": len(v.get("chunks") or []),
        "ranked": [{k: r[k] for k in ("index", "score")} for r in (v.get("ranked") or [])],
        "answer": v.get("answer"),
        "error": v.get("error"),
    }


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    snap = await _graph.aget_state({"configurable": {"thread_id": run_id}})
    if snap is None or not snap.values:
        if run_id in _pending:  # accepted, first checkpoint not yet written
            return _snapshot(run_id, {"status": "running", "steps": []})
        raise HTTPException(404, "run not found")
    return _snapshot(run_id, snap.values)


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    out, seen = [], set()
    async for tup in _saver.alist(None):  # newest-first; dedup on thread
        tid = tup.config["configurable"]["thread_id"]
        if tid in seen:
            continue
        seen.add(tid)
        out.append(_snapshot(tid, tup.checkpoint.get("channel_values") or {}))
        if len(out) >= 50:
            break
    return {"runs": out}
