"""Deterministic LangGraph agent service.

Pipeline: crawl (firecrawl, 2 levels) -> chunk -> embed (ollama) -> retrieve
-> answer, with per-chunk source URLs so answers cite the page they came from.
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
TOP_K = 6
MIN_SIMILARITY = 0.15
ANSWER_TRIES = 3  # single-chunk answer attempts before genuine refusal
CRAWL_LIMIT = int(os.environ.get("AGENT_CRAWL_LIMIT", "15"))  # max pages per run
CRAWL_DEPTH = int(os.environ.get("AGENT_CRAWL_DEPTH", "2"))   # link hops from seed
CRAWL_TIMEOUT_S = float(os.environ.get("AGENT_CRAWL_TIMEOUT", "300"))

http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
_graph = None
_saver = None
_pending: set[str] = set()  # run_ids accepted but not yet checkpointed


class AgentState(TypedDict, total=False):
    url: str
    question: str | None
    pages: list[dict[str, Any]]          # [{url, title, markdown}]
    chunks: list[dict[str, Any]]         # [{text, source}]
    embeddings: list[list[float]]
    ranked: list[dict[str, Any]]         # [{index, score, text, source}]
    sources: list[str]                   # source URLs cited by the answer
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

async def crawl(state: AgentState) -> AgentState:
    """Firecrawl /v2/crawl — async job, follows same-site links CRAWL_DEPTH
    hops from the seed URL up to CRAWL_LIMIT pages."""
    if bail(state):
        return {}
    t0 = time.time()
    try:
        res = await http.post(
            f"{FIRECRAWL}/v2/crawl",
            json={
                "url": state["url"],
                "limit": CRAWL_LIMIT,
                "maxDiscoveryDepth": CRAWL_DEPTH,
                "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
            },
        )
        res.raise_for_status()
        body = res.json()
        job_id = body.get("id") or (body.get("data") or {}).get("id")
        if not body.get("success", True) or not job_id:
            raise RuntimeError(f"crawl submit failed: {str(body)[:200]}")

        deadline = t0 + CRAWL_TIMEOUT_S
        while True:
            await asyncio.sleep(3)
            poll = await http.get(f"{FIRECRAWL}/v2/crawl/{job_id}")
            poll.raise_for_status()
            job = poll.json()
            if job.get("status") == "completed":
                break
            if job.get("status") in ("failed", "cancelled") or time.time() > deadline:
                raise RuntimeError(f"crawl job {job_id} {job.get('status') or 'timed out'}")

        pages = []
        for d in job.get("data") or []:
            md = d.get("markdown") or ""
            if len(md) < 40:
                continue
            meta = d.get("metadata") or {}
            pages.append({
                "url": meta.get("sourceURL") or meta.get("url") or state["url"],
                "title": meta.get("title"),
                "markdown": md,
            })
        if not pages:
            raise RuntimeError("crawl returned no usable pages")
        return {
            "pages": pages,
            "steps": [step("crawl", "ok", ms=round((time.time() - t0) * 1000),
                           pages=len(pages), job=job_id)],
        }
    except Exception as e:
        return fail(state, "crawl", e)


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
    chunks = [
        {"text": t, "source": p["url"]}
        for p in state.get("pages", [])
        for t in _chunk(p["markdown"])
    ]
    if not chunks:
        return fail(state, "chunk", "no usable content after cleaning")
    return {
        "chunks": chunks,
        "steps": [step("chunk", "ok", count=len(chunks), pages=len(state.get("pages") or []))],
    }


async def embed(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    t0 = time.time()
    try:
        res = await http.post(
            f"{OLLAMA}/v1/embeddings",
            json={"model": EMBED_MODEL, "input": [c["text"] for c in state["chunks"]]},
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


# Intent hints: (question pattern, content-marker pattern, boost). Domain-
# agnostic — a cost question deserves chunks that literally contain prices
# on ANY site; embedding similarity alone misses this paraphrase gap.
_INTENTS: list[tuple[re.Pattern, re.Pattern, float]] = [
    (
        re.compile(r"\b(cost|price|pricing|rates?|fees?|charges?|how much|pay)\b", re.I),
        re.compile(r"\$\s?\d|\bper (night|person|couple|guest|hour)|\bfrom \$", re.I),
        0.15,
    ),
    (
        re.compile(r"\b(hours?|open|close|check.?in|check.?out|when)\b", re.I),
        re.compile(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b", re.I),
        0.10,
    ),
]


_STOPWORDS = frozenset(
    "a an the is are was were do does did can could should would will i you we they "
    "it its this that what which who whom how when where why much many any some "
    "of to in on at for with about as by from or and but not no if then than so "
    "be been being have has had having".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _lexical_score(q_terms: set[str], text: str, idf: dict[str, float]) -> float:
    """IDF-weighted coverage: rare question terms (cost, pet) outweigh
    site-ubiquitous ones (the business's own name)."""
    if not q_terms:
        return 0.0
    t_terms = _terms(text)
    return sum(idf[w] for w in q_terms & t_terms) / sum(idf.values())


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

        # Hybrid score over the whole corpus: embedding similarity plus
        # stopword-filtered question-term coverage. The lexical stage
        # rescues chunks whose phrasing diverges from the question but
        # literally name the entity asked about ("$275 per night" barely
        # resembles "how much does it cost" in embedding space).
        q_terms = _terms(q)
        n = len(state["chunks"])
        df = {w: sum(w in _terms(c["text"]) for c in state["chunks"]) for w in q_terms}
        idf = {w: math.log(1 + n / (df[w] or 1)) for w in q_terms}
        intents = [(hint, b) for qr, hint, b in _INTENTS if qr.search(q)]

        def score(c: dict[str, Any], e: list[float]) -> float:
            s = 0.55 * _cos(qv, e) + 0.3 * _lexical_score(q_terms, c["text"], idf)
            return s + sum(b for hint, b in intents if hint.search(c["text"]))

        scored = sorted(
            (
                {
                    "index": i,
                    "score": score(c, e),
                    "text": c["text"],
                    "source": c["source"],
                }
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
    "You are the site's concierge answering a visitor directly, using "
    "ONLY the supplied scraped-page context. Speak plainly in second "
    "person — never mention \"the context\", \"the page\", or that you "
    "are reading from a source. If the context mentions any prices, "
    "packages, offerings, hours, policies, locations, or contacts even "
    "tangentially related to the question, list each one plainly with "
    "its exact qualifiers as written (amounts, units, taxes, minimums, "
    "day or date restrictions, eligibility), and note plainly what "
    "isn't covered. Only if nothing at all is related, say exactly: "
    "\"I don't have that information.\" Never invent facts, dates, "
    "prices, or availability."
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
    # Single-chunk answering: this model size refuses whenever the context
    # mixes chunks that don't all contain the answer — verified empirically.
    # Try the top chunks one at a time; first substantive answer wins and
    # its source is the citation. All-refusal means the corpus genuinely
    # lacks the fact.
    refused = "i don't have that information"
    for r in ranked[:ANSWER_TRIES]:
        try:
            res = await http.post(
                f"{OLLAMA}/v1/chat/completions",
                json={
                    "model": CHAT_MODEL,
                    "temperature": 0,  # deterministic decode
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"CONTEXT:\n[{r['source']}]\n{r['text']}\n\nQUESTION: {state['question']}"},
                    ],
                },
            )
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return fail(state, "answer", e)
        if not text.lower().startswith(refused):
            return {
                "answer": text,
                "sources": [r["source"]],
                "status": "done",
                "steps": [step("answer", "ok", ms=round((time.time() - t0) * 1000), model=CHAT_MODEL, chunk=r["index"])],
            }
    return {
        "answer": "I don't have that information.",
        "status": "done",
        "steps": [step("answer", "ok", grounded=False, tried=min(len(ranked), ANSWER_TRIES))],
    }


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
    for name, fn in [("crawl", crawl), ("chunk", chunk), ("embed", embed), ("retrieve", retrieve), ("answer", answer), ("finish", _finish)]:
        builder.add_node(name, fn)
    builder.add_edge(START, "crawl")
    builder.add_edge("crawl", "chunk")
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
        "title": (v.get("pages") or [{}])[0].get("title"),
        "page_count": len(v.get("pages") or []),
        "steps": v.get("steps", []),
        "chunk_count": len(v.get("chunks") or []),
        "ranked": [{k: r.get(k) for k in ("index", "score", "source")} for r in (v.get("ranked") or [])],
        "sources": v.get("sources") or [],
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
