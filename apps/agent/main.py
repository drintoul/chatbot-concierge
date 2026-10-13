"""Deterministic LangGraph agent service — two checkpointed workflows.

index:  crawl (firecrawl, N levels) -> chunk -> embed -> store corpus
query:  load corpus -> retrieve -> answer

Indexing a site is the slow part (~1 min): crawl, chunk, embed, then persist
the corpus in SQLite. Asking questions is fast (~seconds–40s): retrieval and
a single-chunk LLM call against the stored corpus — no re-crawl per question.

Every edge is fixed code — the LLM never decides control flow. All runs are
checkpointed to SQLite (langgraph-checkpoint-sqlite), so GET /runs/{id}
exposes each node's progression. Determinism: temperature-0 calls,
deterministic chunker, pure-math hybrid ranking, fixed prompts.
"""

import asyncio
import json
import math
import operator
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, TypedDict

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
CRAWL_LIMIT = int(os.environ.get("AGENT_CRAWL_LIMIT", "15"))
CRAWL_DEPTH = int(os.environ.get("AGENT_CRAWL_DEPTH", "2"))
CRAWL_TIMEOUT_S = float(os.environ.get("AGENT_CRAWL_TIMEOUT", "300"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
_index_graph = None
_query_graph = None
_saver: AsyncSqliteSaver = None
_db: aiosqlite.Connection = None  # corpus store (same file as checkpoints)
_pending: set[str] = set()


class AgentState(TypedDict, total=False):
    url: str
    question: str | None
    crawl_limit: int
    crawl_depth: int
    pages: list[dict[str, Any]]          # [{url, title, markdown}]
    phone: str | None                    # best-guess contact number for handoff
    chunks: list[dict[str, Any]]         # [{text, source}]
    embeddings: list[list[float]]
    ranked: list[dict[str, Any]]
    sources: list[str]
    answer: str | None
    # operator.add reducer: every node appends a trace entry.
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


# ------------------------------------------------------- index-graph nodes

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
                "limit": state.get("crawl_limit") or CRAWL_LIMIT,
                "maxDiscoveryDepth": state.get("crawl_depth") or CRAWL_DEPTH,
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


def _canon(url: str) -> str:
    """Corpus key — normalized to a trailing-slash origin."""
    return url.rstrip("/") + "/"


async def store(state: AgentState) -> AgentState:
    """Persist the corpus so query runs reuse it — no re-crawl per question."""
    if bail(state):
        return {}
    url = _canon(state["url"])
    await _db.execute(
        """INSERT INTO corpora(url, title, pages, chunks, embeddings, phone, indexed_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(url) DO UPDATE SET
             title=excluded.title, pages=excluded.pages, chunks=excluded.chunks,
             embeddings=excluded.embeddings, phone=excluded.phone,
             indexed_at=excluded.indexed_at""",
        (
            url,
            (state.get("pages") or [{}])[0].get("title"),
            json.dumps([{k: p[k] for k in ("url", "title")} for p in state.get("pages", [])]),
            json.dumps(state["chunks"]),
            json.dumps(state["embeddings"]),
            state.get("phone"),
            time.time(),
        ),
    )
    await _db.commit()
    return {
        "status": "done",
        "steps": [step("store", "ok", url=url, chunks=len(state["chunks"]))],
    }


# ------------------------------------------------------- query-graph nodes

async def load(state: AgentState) -> AgentState:
    """Pull the persisted corpus for this URL into state."""
    if bail(state):
        return {}
    url = _canon(state["url"])
    async with _db.execute(
        "SELECT chunks, embeddings, phone FROM corpora WHERE url=?", (url,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return fail(state, "load", f"site not indexed — run the Index workflow first: {url}")
    return {
        "chunks": json.loads(row[0]),
        "embeddings": json.loads(row[1]),
        "phone": row[2],
        "steps": [step("load", "ok", url=url)],
    }


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cos(a: list[float], na: float, b: list[float], nb: float) -> float:
    dot = sum(x * y for x, y in zip(a, b))
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


async def retrieve(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    try:
        q = state["question"]
        res = await http.post(
            f"{OLLAMA}/v1/embeddings",
            json={"model": EMBED_MODEL, "input": [q]},
        )
        res.raise_for_status()
        qv = res.json()["data"][0]["embedding"]

        # Hybrid score over the whole corpus: embedding similarity plus
        # stopword-filtered, IDF-weighted question-term coverage. The
        # lexical stage rescues chunks whose phrasing diverges from the
        # question but literally name the entity asked about ("$275 per
        # night" barely resembles "how much does it cost" in embedding
        # space); IDF keeps rare terms (cost, pet) outweighing
        # site-ubiquitous ones (the business's own name).
        chunks, embs = state["chunks"], state["embeddings"]
        q_terms = _terms(q)
        qn = _norm(qv)
        c_terms = [_terms(c["text"]) for c in chunks]
        norms = [_norm(e) for e in embs]
        n = len(chunks)
        df = {w: sum(w in t for t in c_terms) for w in q_terms}
        idf = {w: math.log(1 + n / (df[w] or 1)) for w in q_terms}
        idf_total = sum(idf.values()) or 1
        intents = [(hint, b) for qr, hint, b in _INTENTS if qr.search(q)]

        scored = sorted(
            (
                {
                    "index": i,
                    "score": 0.55 * _cos(qv, qn, e, norms[i])
                    + 0.3 * sum(idf[w] for w in q_terms & c_terms[i]) / idf_total
                    + sum(b for hint, b in intents if hint.search(c["text"])),
                    "text": c["text"],
                    "source": c["source"],
                }
                for i, (c, e) in enumerate(zip(chunks, embs))
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
    "ONLY the supplied scraped-page context. Speak plainly in first "
    "person as the business (\"we\", \"our\") — never mention \"the "
    "context\", \"the page\", or that you are reading from a source. "
    "If the context mentions any prices, packages, offerings, hours, "
    "policies, locations, or contacts even tangentially related to the "
    "question, list each one plainly with its exact qualifiers as "
    "written (amounts, units, taxes, minimums, day or date "
    "restrictions, eligibility), and note plainly what isn't covered. "
    "Only if nothing at all is related, say exactly: \"I don't have "
    "that information.\" Never invent facts, dates, prices, or "
    "availability."
)

# The model's literal refusal is an internal sentinel — the sequential
# loop keys on it, and visitors get a human handoff instead.
REFUSAL = "i don't have that information"

# Soft refusals dodge the sentinel: "we don't have information on X",
# "not mentioned", "I'm sorry, I cannot…". Keying on negation +
# info-words (not the entity) keeps substantive negatives like "No, we
# don't have a bowling alley" from being swallowed.
_SOFT_REFUSAL = re.compile(
    r"\b(don'?t|do not|doesn'?t|does not|cannot|can'?t|couldn'?t|no)\s+"
    r"(have\s+|got\s+)?(enough\s+|any\s+|specific\s+|detailed\s+)?"
    r"(information|info|details?|data|knowledge|mention)\b|"
    r"\b(not|isn'?t|is not|aren'?t|are not|wasn'?t)\s+"
    r"(mentioned|provided|listed|stated|included|specified|found|available)\b|"
    r"\bi('m| am) sorry\b|\bapolog",
    re.I,
)


def _is_refusal(text: str) -> bool:
    # Refusals announce themselves upfront — only scan the opening so a
    # caveat mid-answer ("…but we don't have details on pricing") can't
    # sink an otherwise substantive response.
    head = text[:240].lower()
    return head.startswith(REFUSAL) or bool(_SOFT_REFUSAL.search(head))


def _handoff(phone: str | None) -> str:
    at = f" at {phone}" if phone else ""
    return (
        "I'm sorry — I couldn't find that on our website. "
        f"Please give us a call{at} and our team will be happy to help."
    )


async def answer(state: AgentState) -> AgentState:
    if bail(state):
        return {}
    t0 = time.time()
    ranked = state.get("ranked") or []
    if not ranked:
        return {
            "answer": _handoff(state.get("phone")),
            "status": "done",
            "steps": [step("answer", "ok", grounded=False)],
        }
    # Single-chunk answering: this model size refuses whenever the context
    # mixes chunks that don't all contain the answer — verified empirically.
    # Try the top chunks one at a time; first substantive answer wins and
    # its source is the citation. All-refusal means the corpus genuinely
    # lacks the fact.
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
        if not _is_refusal(text):
            return {
                "answer": text,
                "sources": [r["source"]],
                "status": "done",
                "steps": [step("answer", "ok", ms=round((time.time() - t0) * 1000), model=CHAT_MODEL, chunk=r["index"])],
            }
    return {
        "answer": _handoff(state.get("phone")),
        "status": "done",
        "steps": [step("answer", "ok", grounded=False, tried=min(len(ranked), ANSWER_TRIES))],
    }


# ------------------------------------------------------------------- app

def _build(nodes: list[tuple[str, Any]]) -> Any:
    """Linear chain: START -> nodes in order -> END."""
    builder = StateGraph(AgentState)
    prev = START
    for name, fn in nodes:
        builder.add_node(name, fn)
        builder.add_edge(prev, name)
        prev = name
    builder.add_edge(prev, END)
    return builder.compile(checkpointer=_saver)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index_graph, _query_graph, _saver, _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _saver = AsyncSqliteSaver(await aiosqlite.connect(DB_PATH))
    _db = await aiosqlite.connect(DB_PATH)
    await _db.execute(
        """CREATE TABLE IF NOT EXISTS corpora(
             url TEXT PRIMARY KEY, title TEXT, pages TEXT,
             chunks TEXT, embeddings TEXT, phone TEXT, indexed_at REAL)"""
    )
    await _db.commit()
    # Pre-phone-schema databases: add the column if missing.
    async with _db.execute("PRAGMA table_info(corpora)") as cur:
        cols = {r[1] async for r in cur}
    if "phone" not in cols:
        await _db.execute("ALTER TABLE corpora ADD COLUMN phone TEXT")
        await _db.commit()

    _index_graph = _build(
        [("crawl", crawl), ("chunk", chunk), ("embed", embed), ("store", store)]
    )
    _query_graph = _build([("load", load), ("retrieve", retrieve), ("answer", answer)])
    yield
    await http.aclose()


app = FastAPI(title="hustle-agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IndexRequest(BaseModel):
    url: HttpUrl
    limit: int | None = None  # overrides AGENT_CRAWL_LIMIT when set
    depth: int | None = None  # overrides AGENT_CRAWL_DEPTH when set
    phone: str | None = None  # contact number for handoff answers


class QueryRequest(BaseModel):
    url: HttpUrl
    question: str


async def _run(graph, run_id: str, initial: AgentState) -> None:
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        await graph.ainvoke(
            {**initial, "status": "running", "steps": [step("start", "ok")]},
            cfg,
        )
    except Exception as e:
        try:
            await graph.aupdate_state(cfg, {"status": "failed", "error": str(e)[:300]})
        except Exception:
            pass
    finally:
        _pending.discard(run_id)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/index", status_code=202)
async def start_index(req: IndexRequest) -> dict[str, str]:
    run_id = "idx:" + uuid.uuid4().hex
    _pending.add(run_id)
    initial: AgentState = {"url": str(req.url)}
    if req.limit:
        initial["crawl_limit"] = req.limit
    if req.depth:
        initial["crawl_depth"] = req.depth
    if req.phone:
        initial["phone"] = req.phone.strip() or None
    asyncio.create_task(_run(_index_graph, run_id, initial))
    return {"run_id": run_id, "status": "running"}


@app.post("/query", status_code=202)
async def start_query(req: QueryRequest) -> dict[str, str]:
    run_id = "qry:" + uuid.uuid4().hex
    _pending.add(run_id)
    asyncio.create_task(_run(_query_graph, run_id, {"url": str(req.url), "question": req.question}))
    return {"run_id": run_id, "status": "running"}


@app.get("/sites")
async def list_sites() -> dict[str, Any]:
    out = []
    async with _db.execute(
        "SELECT url, title, pages, json_array_length(chunks), phone, indexed_at"
        " FROM corpora ORDER BY indexed_at DESC"
    ) as cur:
        async for url, title, pages, chunk_count, phone, indexed_at in cur:
            out.append({
                "url": url, "title": title,
                "pages": json.loads(pages),
                "chunk_count": chunk_count,
                "phone": phone,
                "indexed_at": indexed_at,
            })
    return {"sites": out}


def _graph_for(run_id: str):
    return _query_graph if run_id.startswith("qry:") else _index_graph


def _snapshot(run_id: str, v: dict[str, Any]) -> dict[str, Any]:
    v = v or {}
    return {
        "run_id": run_id,
        "status": v.get("status", "running"),
        "url": v.get("url"),
        "question": v.get("question"),
        "title": (v.get("pages") or [{}])[0].get("title"),
        "phone": v.get("phone"),
        "page_count": len(v.get("pages") or []),
        "pages": [{k: p.get(k) for k in ("url", "title")} for p in (v.get("pages") or [])],
        "steps": v.get("steps", []),
        "chunk_count": len(v.get("chunks") or []),
        "ranked": [{k: r.get(k) for k in ("index", "score", "source")} for r in (v.get("ranked") or [])],
        "sources": v.get("sources") or [],
        "answer": v.get("answer"),
        "error": v.get("error"),
    }


@app.get("/runs/{run_id:path}")
async def get_run(run_id: str) -> dict[str, Any]:
    snap = await _graph_for(run_id).aget_state({"configurable": {"thread_id": run_id}})
    if snap is None or not snap.values:
        if run_id in _pending:
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
