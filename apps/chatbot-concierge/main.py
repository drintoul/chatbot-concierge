"""Deterministic LangGraph chatbot concierge service — two checkpointed workflows.

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
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, HttpUrl

OLLAMA = os.environ["OLLAMA_BASE_URL"].rstrip("/")
FIRECRAWL = os.environ["FIRECRAWL_URL"].rstrip("/")
CHAT_MODEL = os.environ["OLLAMA_CHAT_MODEL"]
EMBED_MODEL = os.environ["OLLAMA_EMBED_MODEL"]
DB_PATH = os.environ.get("CHATBOT_CONCIERGE_DB_PATH", "/data/langgraph.db")
TOP_K = 6
MIN_SIMILARITY = 0.15
EVIDENCE_BEST = 0.5   # best-score floor for the no-evidence fast path
ANSWER_TRIES = 3      # single-chunk answer attempts before genuine refusal
CRAWL_LIMIT = int(os.environ.get("CHATBOT_CONCIERGE_CRAWL_LIMIT", "15"))
CRAWL_DEPTH = int(os.environ.get("CHATBOT_CONCIERGE_CRAWL_DEPTH", "2"))
CRAWL_TIMEOUT_S = float(os.environ.get("CHATBOT_CONCIERGE_CRAWL_TIMEOUT", "900"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
_index_graph = None
_query_graph = None
_saver: AsyncSqliteSaver = None
_db: aiosqlite.Connection = None  # corpus store (same file as checkpoints)
_pending: set[str] = set()
_progress: dict[str, dict[str, Any]] = {}  # run_id -> live in-node progress for /runs


class ChatbotConciergeState(TypedDict, total=False):
    url: str
    question: str | None
    crawl_limit: int
    crawl_depth: int
    pages: list[dict[str, Any]]          # [{url, title, markdown}]
    phone: str | None                    # operator-provided contact number
    email: str | None                    # operator-provided contact email
    chunks: list[dict[str, Any]]         # [{text, source}]
    embeddings: list[list[float]]
    ranked: list[dict[str, Any]]
    candidate: str | None               # unverified answer draft
    cand_text: str                      # chunk text the candidate drew from
    cand_source: str
    try_idx: int                        # next ranked chunk to attempt
    grounded: bool                      # verify judged the candidate supported
    missing_terms: list[str]            # question terms absent from whole corpus
    no_evidence: bool                   # all terms absent + weak retrieval
    sources: list[str]
    answer: str | None
    # operator.add reducer: every node appends a trace entry.
    steps: Annotated[list[dict[str, Any]], operator.add]
    status: Literal["running", "done", "failed"]
    error: str | None


def step(node: str, status: str, **detail: Any) -> dict[str, Any]:
    return {"node": node, "status": status, "at": time.time(), **detail}


def fail(state: ChatbotConciergeState, node: str, err: Exception | str) -> ChatbotConciergeState:
    msg = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    return {"status": "failed", "error": msg, "steps": [step(node, "failed", error=msg[:300])]}


def bail(state: ChatbotConciergeState) -> bool:
    return state.get("status") == "failed"


# ------------------------------------------------------- index-graph nodes

async def crawl(state: ChatbotConciergeState, config: RunnableConfig) -> ChatbotConciergeState:
    """Firecrawl /v2/crawl — async job, follows same-site links CRAWL_DEPTH
    hops from the seed URL up to CRAWL_LIMIT pages."""
    if bail(state):
        return {}
    run_id = config["configurable"]["thread_id"]
    limit = state.get("crawl_limit") or CRAWL_LIMIT
    t0 = time.time()
    try:
        res = await http.post(
            f"{FIRECRAWL}/v2/crawl",
            json={
                "url": state["url"],
                "limit": limit,
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
            _progress[run_id] = {
                "node": "crawl",
                "done": job.get("completed") or 0,
                "total": job.get("total") or limit,
            }
            if job.get("status") == "completed":
                break
            if job.get("status") in ("failed", "cancelled") or time.time() > deadline:
                raise RuntimeError(f"crawl job {job_id} {job.get('status') or 'timed out'}")

        # Results paginate at 100 pages — follow `next` until exhausted.
        data = list(job.get("data") or [])
        while job.get("next"):
            nxt = await http.get(
                f"{FIRECRAWL}/v2/crawl/{job_id}", params={"skip": len(data)}
            )
            nxt.raise_for_status()
            job = nxt.json()
            data += job.get("data") or []

        pages = []
        for d in data:
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


async def chunk(state: ChatbotConciergeState) -> ChatbotConciergeState:
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


EMBED_BATCH = 40  # chunks per ollama call — also the progress granularity


async def embed(state: ChatbotConciergeState, config: RunnableConfig) -> ChatbotConciergeState:
    if bail(state):
        return {}
    run_id = config["configurable"]["thread_id"]
    chunks = state["chunks"]
    t0 = time.time()
    try:
        embs = []
        for i in range(0, len(chunks), EMBED_BATCH):
            res = await http.post(
                f"{OLLAMA}/v1/embeddings",
                json={"model": EMBED_MODEL, "input": [c["text"] for c in chunks[i : i + EMBED_BATCH]]},
            )
            res.raise_for_status()
            embs += [d["embedding"] for d in res.json()["data"]]
            _progress[run_id] = {"node": "embed", "done": len(embs), "total": len(chunks)}
        return {
            "embeddings": embs,
            "steps": [step("embed", "ok", ms=round((time.time() - t0) * 1000), count=len(embs), model=EMBED_MODEL)],
        }
    except Exception as e:
        return fail(state, "embed", e)


def _canon(url: str) -> str:
    """Corpus key — normalized to a trailing-slash origin."""
    return url.rstrip("/") + "/"


async def store(state: ChatbotConciergeState) -> ChatbotConciergeState:
    """Persist the corpus so query runs reuse it — no re-crawl per question."""
    if bail(state):
        return {}
    url = _canon(state["url"])
    await _db.execute(
        """INSERT INTO corpora(url, title, pages, chunks, embeddings, phone, email, indexed_at)
           VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(url) DO UPDATE SET
             title=excluded.title, pages=excluded.pages, chunks=excluded.chunks,
             embeddings=excluded.embeddings,
             phone=excluded.phone,
             email=excluded.email,
             indexed_at=excluded.indexed_at""",
        (
            url,
            (state.get("pages") or [{}])[0].get("title"),
            json.dumps([{k: p[k] for k in ("url", "title")} for p in state.get("pages", [])]),
            json.dumps(state["chunks"]),
            json.dumps(state["embeddings"]),
            state.get("phone"),
            state.get("email"),
            time.time(),
        ),
    )
    await _db.commit()
    return {
        "status": "done",
        "steps": [step("store", "ok", url=url, chunks=len(state["chunks"]))],
    }


# ------------------------------------------------------- query-graph nodes

async def load(state: ChatbotConciergeState) -> ChatbotConciergeState:
    """Pull the persisted corpus for this URL into state."""
    if bail(state):
        return {}
    url = _canon(state["url"])
    async with _db.execute(
        "SELECT chunks, embeddings, phone, email FROM corpora WHERE url=?", (url,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return fail(state, "load", f"site not indexed — run the Index workflow first: {url}")
    return {
        "chunks": json.loads(row[0]),
        "embeddings": json.loads(row[1]),
        "phone": row[2],
        "email": row[3],
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


async def retrieve(state: ChatbotConciergeState) -> ChatbotConciergeState:
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
        # No-evidence flag: every content term absent from the corpus AND
        # weak retrieval — a vocabulary disconnect that can't be answered.
        # Requires ALL terms missing: inflections ("used" vs "use") make
        # any-single-term-missing too trigger-happy.
        no_evidence = bool(
            q_terms
            and all(df[w] == 0 for w in q_terms)
            and (scored[0]["score"] < EVIDENCE_BEST if scored else False)
        )
        return {
            "ranked": ranked,
            "missing_terms": sorted(w for w in q_terms if df[w] == 0),
            "no_evidence": no_evidence,
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
    r"(have\s+|got\s+)?(enough\s+|any\s+|specific\s+|detailed\s+|that\s+|the\s+)?"
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


def _handoff(phone: str | None, email: str | None) -> str:
    if phone and email:
        contact = f"Please give us a call at {phone} or email us at {email}"
    elif phone:
        contact = f"Please give us a call at {phone}"
    elif email:
        contact = f"Please email us at {email}"
    else:
        contact = "Please give us a call"
    return (
        "I'm sorry — I couldn't find that on our website. "
        f"{contact} and our team will be happy to help."
    )


async def answer(state: ChatbotConciergeState) -> ChatbotConciergeState:
    """Draft a candidate from the next ranked chunk — single-chunk
    contexts are the only ones this model size handles reliably
    (verified empirically). It does NOT ship: verify decides that."""
    if bail(state):
        return {}
    t0 = time.time()
    ranked = state.get("ranked") or []
    try_idx = state.get("try_idx", 0)
    # Fast miss: the question's entire vocabulary is absent from the
    # corpus and retrieval is weak — no chunk can support an answer, so
    # skip the decode loop (each attempt is a ~30s+ LLM call) and let
    # verify emit the handoff.
    if try_idx == 0 and state.get("no_evidence"):
        return {
            "candidate": None,
            "steps": [step("answer", "ok", skipped=True, missing=state.get("missing_terms") or [])],
        }
    if not ranked or try_idx >= min(len(ranked), ANSWER_TRIES):
        return {"candidate": None, "steps": [step("answer", "ok", tried=try_idx)]}
    r = ranked[try_idx]
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
    return {
        "candidate": text,
        "cand_text": r["text"],
        "cand_source": r["source"],
        "try_idx": try_idx + 1,
        "steps": [step("answer", "ok", ms=round((time.time() - t0) * 1000), model=CHAT_MODEL, chunk=r["index"], attempt=try_idx + 1)],
    }


# Grounding pass: every factual claim in the candidate must appear in
# the source chunk. Unverifiable content — including absence claims the
# page doesn't explicitly state — is rejected, never shipped.
VERIFY_PROMPT = (
    "You verify whether an answer is supported by a source text. Reply "
    "with exactly one word: YES only if every factual claim in the "
    "answer appears in the source (paraphrase is fine — amounts, dates, "
    "hours, policies, amenities, contacts must match). Reply NO if any "
    "claim is missing from, contradicted by, or goes beyond the source. "
    "A claim that something does NOT exist is YES only when the source "
    "explicitly says so — absence of mention is not support."
)


async def verify(state: ChatbotConciergeState) -> ChatbotConciergeState:
    """Decide whether the candidate ships, retries, or becomes a
    handoff. Terminates the loop by setting status=done."""
    if bail(state):
        return {}
    t0 = time.time()
    cand = state.get("candidate")
    ranked = state.get("ranked") or []
    tries_left = state.get("try_idx", 0) < min(len(ranked), ANSWER_TRIES)

    def ungrounded(**detail: Any) -> ChatbotConciergeState:
        out: ChatbotConciergeState = {
            "grounded": False,
            "steps": [step("verify", "ok", ms=round((time.time() - t0) * 1000), grounded=False, **detail)],
        }
        if not tries_left:
            out.update(answer=_handoff(state.get("phone"), state.get("email")), status="done")
        return out

    if not cand:
        # No candidate at all — nothing left to try (empty ranked or the
        # no-evidence fast path). Terminal: always hand off here.
        return {
            "answer": _handoff(state.get("phone"), state.get("email")),
            "status": "done",
            "grounded": False,
            "steps": [step("verify", "ok", ms=round((time.time() - t0) * 1000), grounded=False)],
        }
    if _is_refusal(cand):
        return ungrounded(reason="refusal")
    try:
        res = await http.post(
            f"{OLLAMA}/v1/chat/completions",
            json={
                "model": CHAT_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": VERIFY_PROMPT},
                    {"role": "user", "content": f"SOURCE:\n{state['cand_text']}\n\nANSWER:\n{cand}"},
                ],
            },
        )
        res.raise_for_status()
        verdict = res.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        return fail(state, "verify", e)
    if not verdict.startswith("yes"):
        return ungrounded(verdict=verdict[:60])
    return {
        "answer": cand,
        "sources": [state["cand_source"]],
        "status": "done",
        "grounded": True,
        "steps": [step("verify", "ok", ms=round((time.time() - t0) * 1000), grounded=True)],
    }


# ------------------------------------------------------------------- app

def _build(nodes: list[tuple[str, Any]], retry: str | None = None) -> Any:
    """Linear chain: START -> nodes in order. With retry, the last node
    gets a conditional edge back to `retry` until status is terminal."""
    builder = StateGraph(ChatbotConciergeState)
    prev = START
    for name, fn in nodes:
        builder.add_node(name, fn)
        builder.add_edge(prev, name)
        prev = name
    if retry:
        builder.add_conditional_edges(
            prev, lambda s: END if s.get("status") in ("done", "failed") else retry
        )
    else:
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
             chunks TEXT, embeddings TEXT, phone TEXT, email TEXT,
             indexed_at REAL)"""
    )
    await _db.execute(
        """CREATE TABLE IF NOT EXISTS events(
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             ts REAL, run_id TEXT, url TEXT, question TEXT, answer TEXT,
             grounded INTEGER, sources TEXT, latency_ms INTEGER, error TEXT)"""
    )
    await _db.commit()
    # Older databases: add any columns introduced after first boot.
    async with _db.execute("PRAGMA table_info(corpora)") as cur:
        cols = {r[1] async for r in cur}
    for col in ("phone", "email"):
        if col not in cols:
            await _db.execute(f"ALTER TABLE corpora ADD COLUMN {col} TEXT")
    await _db.commit()

    _index_graph = _build(
        [("crawl", crawl), ("chunk", chunk), ("embed", embed), ("store", store)]
    )
    _query_graph = _build(
        [("load", load), ("retrieve", retrieve), ("answer", answer), ("verify", verify)],
        retry="answer",
    )
    yield
    await http.aclose()


app = FastAPI(title="chatbot-concierge", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IndexRequest(BaseModel):
    url: HttpUrl
    limit: int | None = None  # overrides CHATBOT_CONCIERGE_CRAWL_LIMIT when set
    depth: int | None = None  # overrides CHATBOT_CONCIERGE_CRAWL_DEPTH when set
    phone: str | None = None  # contact number for handoff answers
    email: str | None = None  # contact email for handoff answers


class QueryRequest(BaseModel):
    url: HttpUrl
    question: str


async def _audit(graph, run_id: str, cfg: dict[str, Any]) -> None:
    """Persist each finished query run — question, response, latency.
    Never raises: auditing must not break a run."""
    if not run_id.startswith("qry:"):
        return
    try:
        snap = await graph.aget_state(cfg)
        v = (snap.values if snap else {}) or {}
        steps = v.get("steps") or []
        latency = round((steps[-1]["at"] - steps[0]["at"]) * 1000) if len(steps) >= 2 else None
        await _db.execute(
            "INSERT INTO events(ts, run_id, url, question, answer, grounded,"
            " sources, latency_ms, error) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                time.time(), run_id, v.get("url"), v.get("question"),
                v.get("answer"),
                (1 if v.get("grounded") else 0) if v.get("status") == "done" else None,
                json.dumps(v.get("sources") or []),
                latency, v.get("error"),
            ),
        )
        await _db.commit()
    except Exception:
        pass


async def _run(graph, run_id: str, initial: ChatbotConciergeState) -> None:
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
        _progress.pop(run_id, None)
        await _audit(graph, run_id, cfg)


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
    initial: ChatbotConciergeState = {"url": str(req.url)}
    if req.limit:
        initial["crawl_limit"] = req.limit
    if req.depth:
        initial["crawl_depth"] = req.depth
    if req.phone:
        initial["phone"] = req.phone.strip() or None
    if req.email:
        initial["email"] = req.email.strip() or None
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
        "SELECT url, title, pages, json_array_length(chunks), phone, email, indexed_at"
        " FROM corpora ORDER BY indexed_at DESC"
    ) as cur:
        async for url, title, pages, chunk_count, phone, email, indexed_at in cur:
            out.append({
                "url": url, "title": title,
                "pages": json.loads(pages),
                "chunk_count": chunk_count,
                "phone": phone,
                "email": email,
                "indexed_at": indexed_at,
            })
    return {"sites": out}


@app.delete("/sites")
async def delete_site(url: str) -> dict[str, Any]:
    """Remove a corpus and its audit events."""
    canon = _canon(url)
    async with _db.execute("DELETE FROM corpora WHERE url=?", (canon,)) as cur:
        corpus = cur.rowcount
    async with _db.execute("DELETE FROM events WHERE url=?", (canon,)) as cur:
        evts = cur.rowcount
    await _db.commit()
    return {"deleted": corpus > 0, "events_deleted": evts}


@app.get("/audit")
async def list_audit(limit: int = 50, url: str | None = None) -> dict[str, Any]:
    out = []
    sql = (
        "SELECT ts, run_id, url, question, answer, grounded, sources,"
        " latency_ms, error FROM events"
    )
    args: list[Any] = []
    if url:
        sql += " WHERE url=?"
        args.append(_canon(url))
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(min(max(limit, 1), 200))
    async with _db.execute(sql, args) as cur:
        async for ts, run_id, url, question, answer, grounded, sources, lat, err in cur:
            out.append({
                "ts": ts, "run_id": run_id, "url": url, "question": question,
                "answer": answer, "grounded": grounded,
                "sources": json.loads(sources or "[]"),
                "latency_ms": lat, "error": err,
            })
    return {"events": out}


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
        "email": v.get("email"),
        "page_count": len(v.get("pages") or []),
        "pages": [{k: p.get(k) for k in ("url", "title")} for p in (v.get("pages") or [])],
        "steps": v.get("steps", []),
        "chunk_count": len(v.get("chunks") or []),
        "ranked": [{k: r.get(k) for k in ("index", "score", "source")} for r in (v.get("ranked") or [])],
        "sources": v.get("sources") or [],
        "answer": v.get("answer"),
        "grounded": v.get("grounded"),
        "error": v.get("error"),
        "progress": _progress.get(run_id),
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
