"""Deterministic LangGraph chatbot concierge service — query-only.

query:  load corpus -> retrieve -> answer -> verify (loop until grounded)

The knowledge store (the `corpora` table) is built ahead of release by the
separate `extractor` project — firecrawl/playwright crawl, admin-built
URLs, chunk -> embed -> store. This service never crawls or re-indexes, so
the deployed corpus can't be modified by visitors. Asking questions is fast
(~seconds–40s): retrieval and single-chunk LLM calls against the stored
corpus — no re-crawl per question.

Every edge is fixed code — the LLM never decides control flow. All runs are
checkpointed to SQLite (langgraph-checkpoint-sqlite), so GET /runs/{id}
exposes each node's progression. Determinism: temperature-0 calls,
pure-math hybrid ranking, fixed prompts.
"""

import asyncio
import json
import math
import operator
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import urlparse

import aiosqlite
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fpdf import FPDF
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, HttpUrl

OLLAMA = os.environ["OLLAMA_BASE_URL"].rstrip("/")
QDRANT = os.environ.get("QDRANT_URL", "http://rag-qdrant:6333").rstrip("/")
CHAT_MODEL = os.environ["OLLAMA_CHAT_MODEL"]
EMBED_MODEL = os.environ["OLLAMA_EMBED_MODEL"]
DB_PATH = os.environ.get("CHATBOT_CONCIERGE_DB_PATH", "/data/langgraph.db")
# Business-local timezone for resolving "tonight/this weekend" — containers
# run UTC, which is already tomorrow for Pacific evenings. Service-level
# default; each corpus can carry its own `tz` set by the extractor.
TZ = ZoneInfo(os.environ.get("TZ", "UTC"))

TOP_K = 6
MIN_SIMILARITY = 0.15
EVIDENCE_BEST = 0.5   # best-score floor for the no-evidence fast path
ANSWER_TRIES = 6      # single-chunk answer attempts — covers all of TOP_K
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
_query_graph = None
_saver: AsyncSqliteSaver = None
_db: aiosqlite.Connection = None  # corpus store (same file as checkpoints)
_pending: set[str] = set()
_progress: dict[str, dict[str, Any]] = {}  # run_id -> live in-node progress for /runs


class ChatbotConciergeState(TypedDict, total=False):
    url: str
    question: str | None
    pages: list[dict[str, Any]]          # [{url, title}] — snapshot detail for old runs
    phone: str | None                    # contact number for handoff answers
    email: str | None                    # contact email for handoff answers
    tz: str | None                       # business-local IANA zone for "today"
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


def _canon(url: str) -> str:
    """Corpus key — normalized to a trailing-slash origin."""
    return url.rstrip("/") + "/"


# Widget keys are Canada-flavoured words — 8+ chars, easy to say over the
# phone, nothing confrontational. Small keyspace vs random glyphs (~160
# words), fine for dev/demo; production should rate-limit /query regardless.
_TOKEN_WORDS = (
    # BC
    "VANCOUVER WHISTLER VICTORIA OKANAGAN CHILLIWACK ABBOTSFORD RICHMOND "
    "COQUITLAM PORTCOQUITLAM NEWWESTMINSTER LILLOOET PEMBERTON POWELLRIVER "
    "COURTENAY CAMPBELLRIVER PORTALBERNI PORTHARDY PRINCERUPERT PRINCEGEORGE "
    "FORTSTJOHN CRANBROOK CASTLEGAR REVELSTOKE PENTICTON SALMONARM SMITHERS "
    "WILLIAMSLAKE HUNDREDMILE DAWSONCREEK FORTNELSON PARKSVILLE QUALICUM "
    "UCLUELET TELEGRAPHCOVE HAIDAGWAII SALTSPRING SUNSHINECOAST SEATOSKY "
    "CHILCOTIN KOOTENAYS CASCADES PACIFICRIM STRATHCONA GARIBALDI MTROBSON "
    "COLUMBIA THOMPSON SIMILKAMEEN ARROWLAKES INVERMERE KIMBERLEY SPARWOOD "
    "GRANDFORKS MAPLERIDGE PITTMEADOWS WHITEROCK TSAWWASSEN HORSESHOE DEEPCOVE "
    "LYNNVALLEY CAPILANO MTSEYMOUR MTWASHINGTON BIGWHITE SILVERSTAR SUNPEAKS "
    "MANNINGPARK CATHEDRAL STAWAMUS BRANDYWINE BIRKENHEAD ANDERSON BOSTONBAR "
    "ASHCROFT CACHECREEK SPENCESBRIDGE LOGANLAKE BARRIERE CLEARWATER VALEMOUNT "
    "MACKENZIE CHETWYND HUDSONHOPE TUMBLERRIDGE SALISHSEA SPIRITBEAR BALDEAGLE "
    "REDCEDAR HUCKLEBERRY SASQUATCH "
    # Rest of Canada — cities, provinces, landmarks, Canadiana
    "MONTREAL EDMONTON WINNIPEG SASKATOON FREDERICTON CHARLOTTETOWN "
    "WHITEHORSE YELLOWKNIFE KITCHENER HAMILTON GATINEAU SHERBROOKE "
    "TROISRIVIERES SAINTJOHN CAPEBRETON LABRADOR CHURCHILL TUKTOYAKTUK "
    "DAWSONCITY WATSONLAKE MANITOBA NOVASCOTIA NEWBRUNSWICK NEWFOUNDLAND "
    "ROCKYMOUNTAINS LAKELOUISE MORAINELAKE NIAGARAFALLS ALGONQUIN GROSMORNE "
    "BAYOFFUNDY NORTHERNLIGHTS PRAIRIES MARITIMES GREATLAKES SUPERIOR "
    "ATHABASCA FRASERRIVER OTTAWARIVER STLAWRENCE REDRIVER MAPLELEAF "
    "MAPLESYRUP BUTTERTART NANAIMOBAR TOURTIERE KETCHUPCHIPS MOUNTIES "
    "LACROSSE DOUBLEDOUBLE LUMBERJACK VOYAGEUR POLARBEAR ATLANTIC CANADAGOOSE"
).split()
_TOKEN_RE = re.compile(r"[A-Z]{8,}")


def _new_token() -> str:
    return secrets.choice(_TOKEN_WORDS)


async def _unused_token() -> str:
    """Word keyspace is small — always roll against existing tokens."""
    async with _db.execute("SELECT token FROM corpora") as cur:
        used = {r[0] async for r in cur}
    while True:
        t = _new_token()
        if t not in used:
            used.add(t)
            return t


def _site_title(state: ChatbotConciergeState) -> str | None:
    """Site title from the homepage page, not whichever page listed first."""
    pages = state.get("pages") or []
    seed = _canon(state.get("url", "")).rstrip("/")
    for p in pages:
        if (p.get("url") or "").rstrip("/") == seed:
            return p.get("title")
    return (pages[0] if pages else {}).get("title")


# ------------------------------------------------------- query-graph nodes

async def load(state: ChatbotConciergeState) -> ChatbotConciergeState:
    """Pull the persisted corpus for this URL into state."""
    if bail(state):
        return {}
    url = _canon(state["url"])
    async with _db.execute(
        "SELECT chunks, embeddings, phone, email, tz FROM corpora WHERE url=?", (url,)
    ) as cur:
        row = await cur.fetchone()
    if not row:
        return fail(state, "load", f"site not indexed — build its corpus with the extractor project: {url}")
    return {
        "chunks": json.loads(row[0]),
        "embeddings": json.loads(row[1]),
        "phone": row[2],
        "email": row[3],
        "tz": row[4],
        "steps": [step("load", "ok", url=url)],
    }


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cos(a: list[float], na: float, b: list[float], nb: float) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb) if na and nb else 0.0


_HOURS_Q = re.compile(r"\b(hours?|open|close|check.?in|check.?out|when)\b", re.I)

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
        _HOURS_Q,
        re.compile(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b|\b(mon|tue|wed|thu|fri|sat|sun)(day)?\b", re.I),
        0.10,
    ),
    (
        re.compile(r"\b(where|location|address|directions|located|find|visit|map)\b", re.I),
        re.compile(r"\b\d{2,5}\s+[a-z0-9 .]+?\b(rd|st|ave|blvd|dr|ln|lane|way|hwy|highway|road|street|avenue|drive)\b\.?", re.I),
        0.25,
    ),
    (
        # Menu questions → chunks that actually read like a menu (beer
        # specs, dish items, prices) rather than about-us prose.
        re.compile(r"\b(beers?|ales?|ipa|lager|stout|porter|pilsner|drinks?|cocktails?|wine|menu|foods?|meals?|dish|dishes|pizzas?|kabobs?|kebabs?|entrees?|appetizers?|desserts?|popular|favou?rite|recommend|best)\b", re.I),
        re.compile(r"\b(abv|ibu|\d+(\.\d+)?\s?%\s?(alc|abv)|on tap|pints?|flight|growler|marinated|toppings?|skewers?|served with|with rice|garlic|pepperoni)\b", re.I),
        0.15,
    ),
]


# Pages that carry canonical business facts — when an intent fires, chunks
# from these paths outrank incidental matches on event/blog pages, which are
# also full of times/dates/prices but answer a different question.
# Pages whose nav/footer sections commonly hold second-level utility links
# (hours, closures, policies, directions) that the homepage doesn't surface.
_HUB_PATH = re.compile(
    r"^/(contact|about|services?|info|visit|hours?|locations?|find-?us|"
    r"help|support|faqs?|company)([/_-]|$)",
    re.I,
)


_CORE_PATH = re.compile(
    r"^/(contact|about|hours?|find-?us|visit|location|directions|menu|"
    r"rates?|pric|book|reserv|stay|rooms?|accommodation|faq|info|"
    r"where-to|polic)([/_-]|$)",
    re.I,
)
CORE_BOOST = 0.12

# Mirror of _CORE_PATH: recurring-content paths (events, blogs, galleries,
# area pages) are full of times/dates/prices but answer a *different*
# question. On intent questions they get pushed down — unless the question
# itself is about events ("when is trivia night" SHOULD find event pages).
_SERIES_PATH = re.compile(
    r"^/(events?|good-?times|blogs?|posts?|news|calendar|promos?|"
    r"galler|areas|shows?|specials?)([/_-]|$)",
    re.I,
)
SERIES_PENALTY = 0.10
_EVENTY_Q = re.compile(
    r"\b(events?|trivia|fests?|concerts?|shows?|nights?|games?|"
    r"live music|happening|going on|schedule|calendar)\b",
    re.I,
)

# Signature of an actual hours block: repeated weekday→clock-time pairs
# ("Saturday 10am to 1am"). A daily-specials or event chunk can contain a
# stray day+time ("Thursday (After 5 PM)") — it takes ≥3 pairs to earn the
# boost, which incidental matches don't produce.
_DAY_TIME = re.compile(
    r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b[^\n]{0,15}?\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
    re.I,
)
_HOURS_TABLE_MIN = 3
HOURS_BLOCK_BOOST = 0.14

# Price lists are the classic hours-question impostor: daily specials and
# menus are dense with day names AND clock times AND the word "open" ("Happy
# Hour pricing from open to close"), but a real hours table has no prices.
# On hours questions, demote currency-dense chunks — gated by the same
# eventy guard as SERIES_PENALTY since legit event pages carry prices too.
_PRICE = re.compile(r"\$\s?\d|\d+\s?¢")
_PRICE_DENSE_MIN = 3
PRICE_LIST_PENALTY = 0.12


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
        chunks = state["chunks"]
        q_terms = _terms(q)
        qn = _norm(qv)

        # Fast path: the shared qdrant holds corpus vectors — take its
        # top-30 hits and re-rank them through the lexical layer below.
        # Corpora built before qdrant (no points) or an outage fall back
        # to in-DB cosine over the full chunk list.
        vec = None
        try:
            r = await http.post(
                f"{QDRANT}/collections/corpora/points/search",
                json={
                    "vector": qv, "limit": 30, "with_payload": True,
                    "filter": {"must": [
                        {"key": "site", "match": {"value": state["url"]}},
                    ]},
                },
            )
            if r.status_code == 200 and r.json().get("result"):
                hits = r.json()["result"]
                chunks = [{"text": h["payload"]["text"],
                           "source": h["payload"]["source"]} for h in hits]
                vec = [h["score"] for h in hits]
        except Exception:
            pass
        if vec is None:
            vec = [_cos(qv, qn, e, _norm(e)) for e in state["embeddings"]]
        c_terms = [_terms(c["text"]) for c in chunks]
        n = len(chunks)
        df = {w: sum(w in t for t in c_terms) for w in q_terms}
        idf = {w: math.log(1 + n / (df[w] or 1)) for w in q_terms}
        idf_total = sum(idf.values()) or 1
        intents = [(hint, b) for qr, hint, b in _INTENTS if qr.search(q)]
        eventy_q = bool(_EVENTY_Q.search(q))
        hours_q = bool(_HOURS_Q.search(q))
        core_hit = [
            bool(intents) and bool(_CORE_PATH.search(urlparse(c["source"]).path or ""))
            for c in chunks
        ]
        series_hit = [
            bool(intents) and not eventy_q
            and bool(_SERIES_PATH.search(urlparse(c["source"]).path or ""))
            for c in chunks
        ]
        hours_block = [
            hours_q and len(_DAY_TIME.findall(c["text"])) >= _HOURS_TABLE_MIN
            for c in chunks
        ]
        price_list = [
            hours_q and not eventy_q
            and len(_PRICE.findall(c["text"])) >= _PRICE_DENSE_MIN
            for c in chunks
        ]

        scored = sorted(
            (
                {
                    "index": i,
                    "score": 0.55 * vec[i]
                    + 0.3 * sum(idf[w] for w in q_terms & c_terms[i]) / idf_total
                    + sum(b for hint, b in intents if hint.search(c["text"]))
                    + (CORE_BOOST if core_hit[i] else 0.0)
                    - (SERIES_PENALTY if series_hit[i] else 0.0)
                    + (HOURS_BLOCK_BOOST if hours_block[i] else 0.0)
                    - (PRICE_LIST_PENALTY if price_list[i] else 0.0),
                    "text": c["text"],
                    "source": c["source"],
                }
                for i, c in enumerate(chunks)
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
    "context\", \"the page\", \"the review\", \"the website\", or that "
    "you are reading from a source. When context describes a visitor or "
    "third party, refer to them directly (\"a guest caught…\"), not via "
    "source narration (\"the review says…\"). "
    "If the context mentions any prices, packages, offerings, hours, "
    "policies, locations, or contacts even tangentially related to the "
    "question, list each one plainly with its exact qualifiers as "
    "written (amounts, units, taxes, minimums, day or date "
    "restrictions, eligibility), and note plainly what isn't covered. "
    "When the question isn't literally answered — a superlative like "
    "\"most popular\" or \"best\", a recommendation, or a specific item — "
    "give the closest related offerings the context does list and note "
    "the missing part (\"we don't publish a most-popular list, but our "
    "lineup includes…\") — never start that reply with \"I don't have "
    "that information.\" Save \"I don't have that information.\" as the "
    "entire reply for when nothing at all relates. Never invent "
    "facts, dates, prices, or availability."
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
    # caveat mid-answer can't sink a substantive response. A soft apology
    # only counts when it's the whole reply: a long answer that *also*
    # apologizes ("sorry, we don't publish X, but our lineup includes…")
    # still carries real content.
    head = text[:240].lower()
    if head.startswith(REFUSAL):
        # The sentinel only counts as a refusal when it IS the reply —
        # the model also uses it as preamble to a closest-info answer.
        return len(text) < 200
    return len(text) < 200 and bool(_SOFT_REFUSAL.search(head))


def _site_tz(state: ChatbotConciergeState) -> ZoneInfo:
    """The corpus's stored zone, else the service default. Bad values can't
    reach here (validated at index time), but stay safe anyway."""
    try:
        return ZoneInfo(state["tz"]) if state.get("tz") else TZ
    except Exception:
        return TZ


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
                "think": False,   # qwen3 — keep reasoning off the answer path
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    # TODAY pins "tonight/today/tomorrow/this weekend" to a
                    # real weekday — without it the model guesses one and the
                    # verify pass can't catch the invented premise. Per-site
                    # tz falls back to the service-level TZ.
                    {"role": "user", "content": f"TODAY: {datetime.now(_site_tz(state)):%A, %B %-d, %Y}\n\nCONTEXT:\n[{r['source']}]\n{r['text']}\n\nQUESTION: {state['question']}"},
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
    "hours, policies, amenities, contacts must match) AND the source is "
    "actually about the subject the question asks. Information about a "
    "different business, place, event, or entity is not support even "
    "when the claims match it — a museum's winter hours don't answer "
    "a question about the lodge. Reply NO if any claim is missing "
    "from, contradicted by, or goes beyond the source, or if the "
    "source answers a different subject. A claim that something does "
    "NOT exist is YES only when the source explicitly says so — "
    "absence of mention is not support."
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
                "think": False,   # qwen3 — keep reasoning off the verify path
                "messages": [
                    {"role": "system", "content": VERIFY_PROMPT},
                    {"role": "user", "content": f"QUESTION: {state['question']}\n\nSOURCE:\n{state['cand_text']}\n\nANSWER:\n{cand}"},
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
    global _query_graph, _saver, _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _saver = AsyncSqliteSaver(await aiosqlite.connect(DB_PATH))
    _db = await aiosqlite.connect(DB_PATH)
    await _db.execute(
        """CREATE TABLE IF NOT EXISTS corpora(
             url TEXT PRIMARY KEY, title TEXT, pages TEXT,
             chunks TEXT, embeddings TEXT, phone TEXT, email TEXT,
             token TEXT, widget TEXT, indexed_at REAL)"""
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
    for col, typ in {"phone": "TEXT", "email": "TEXT",
                     "token": "TEXT", "widget": "TEXT", "tz": "TEXT"}.items():
        if col not in cols:
            await _db.execute(f"ALTER TABLE corpora ADD COLUMN {col} {typ}")
    await _db.commit()
    # Rows with no token (or an old charset one) get a fresh word.
    async with _db.execute("SELECT url, token FROM corpora") as cur:
        stale = [r[0] async for r in cur if not r[1] or not _TOKEN_RE.fullmatch(r[1])]
    for u in stale:
        await _db.execute(
            "UPDATE corpora SET token=? WHERE url=?", (await _unused_token(), u)
        )
    if stale:
        await _db.commit()

    _query_graph = _build(
        [("load", load), ("retrieve", retrieve), ("answer", answer), ("verify", verify)],
        retry="answer",
    )
    yield
    await http.aclose()


app = FastAPI(title="chatbot-concierge", lifespan=lifespan)
# Widget embeds post cross-origin. Wide-open for local dev; tighten to
# per-tenant origin allowlists before public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QueryRequest(BaseModel):
    url: HttpUrl
    question: str
    token: str | None = None  # widget key — must match the corpus's stored token


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


@app.post("/query", status_code=202)
async def start_query(req: QueryRequest) -> dict[str, str]:
    url = _canon(str(req.url))
    token = (req.token or "").strip().upper()
    async with _db.execute(
        "SELECT token FROM corpora WHERE url=?", (url,)
    ) as cur:
        row = await cur.fetchone()
    if row and token != row[0]:
        raise HTTPException(403, "invalid token for this site")
    run_id = "qry:" + uuid.uuid4().hex
    _pending.add(run_id)
    asyncio.create_task(_run(_query_graph, run_id, {"url": str(req.url), "question": req.question}))
    return {"run_id": run_id, "status": "running"}


class WidgetConfig(BaseModel):
    url: HttpUrl
    name: str | None = None
    greeting: str | None = None
    colour: str | None = None


@app.post("/widget-config")
async def save_widget_config(req: WidgetConfig) -> dict[str, bool]:
    """Persist per-site widget theming so the Chatbot tab and generated
    snippet reload it instead of being re-entered each visit."""
    url = _canon(str(req.url))
    cfg = {
        k: v.strip()
        for k, v in {"name": req.name, "greeting": req.greeting,
                     "colour": req.colour}.items()
        if v and v.strip()
    }
    cur = await _db.execute(
        "UPDATE corpora SET widget=? WHERE url=?", (json.dumps(cfg), url)
    )
    await _db.commit()
    if not cur.rowcount:
        raise HTTPException(404, "site not indexed")
    return {"ok": True}


@app.get("/widget-config")
async def widget_config(token: str) -> dict[str, Any]:
    """The embed tag carries only ?token= — the widget calls this to resolve
    its site URL and saved theming. Token is the lookup key, so nothing else
    needs to be in the tag."""
    async with _db.execute(
        "SELECT url, widget, phone, email FROM corpora WHERE token=?",
        (token.strip().upper(),),
    ) as cur:
        row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "unknown token")
    cfg = json.loads(row[1]) if row[1] else {}
    out = {"url": row[0], **cfg}
    # Contact details only leave the corpus when the admin opted the widget in.
    if cfg.get("contact"):
        if row[2]:
            out["phone"] = row[2]
        if row[3]:
            out["email"] = row[3]
    return out


@app.get("/sites/screenshot")
async def site_screenshot(token: str) -> Response:
    """Homepage PNG captured at indexing time — read-only from the corpus."""
    async with _db.execute(
        "SELECT screenshot FROM corpora WHERE token=?", (token.strip().upper(),)
    ) as cur:
        row = await cur.fetchone()
    if not row or not row[0]:
        raise HTTPException(404, "no screenshot for this site")
    return Response(
        bytes(row[0]),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/sites")
async def list_sites() -> dict[str, Any]:
    out = []
    async with _db.execute(
        "SELECT url, title, pages, json_array_length(chunks), phone, email, token,"
        " widget, tz, indexed_at FROM corpora ORDER BY indexed_at DESC"
    ) as cur:
        async for url, title, pages, chunk_count, phone, email, token, widget, tz, indexed_at in cur:
            out.append({
                "url": url, "title": title,
                "pages": json.loads(pages),
                "chunk_count": chunk_count,
                "phone": phone,
                "email": email,
                "token": token,
                "widget": json.loads(widget) if widget else {},
                "tz": tz,
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
    try:
        await http.post(
            f"{QDRANT}/collections/corpora/points/delete",
            json={"filter": {"must": [{"key": "site", "match": {"value": canon}}]}},
        )
    except Exception:
        pass
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


@app.delete("/audit")
async def clear_audit(url: str | None = None) -> dict[str, int]:
    """Clear audit events — one site when url is given, all sites otherwise."""
    if url:
        async with _db.execute(
            "DELETE FROM events WHERE url=?", (_canon(url),)
        ) as cur:
            n = cur.rowcount
    else:
        async with _db.execute("DELETE FROM events") as cur:
            n = cur.rowcount
    await _db.commit()
    return {"deleted": n}


_PDF_MAP = str.maketrans({"→": "->", "’": "'", "‘": "'", "“": '"', "”": '"',
                          "–": "-", "—": "-", "…": "...", "·": "-"})


def _pdf_text(s: str | None) -> str:
    """Core PDF fonts are latin-1 only — fold unicode, replace the rest,
    and break >60-char unbreakable runs (URLs) so word-wrap never overflows."""
    t = (s or "").translate(_PDF_MAP).encode("latin-1", "replace").decode("latin-1")
    return re.sub(r"\S{60}", lambda m: m.group(0) + " ", t)


@app.get("/audit/pdf")
async def audit_pdf(url: str | None = None) -> Response:
    """Download the audit log as a PDF — filtered by site when ?url= is given."""
    sql = (
        "SELECT ts, run_id, url, question, answer, grounded, sources,"
        " latency_ms, error FROM events"
    )
    args: list[Any] = []
    host = "all-sites"
    if url:
        canon = _canon(url)
        sql += " WHERE url=?"
        args.append(canon)
        host = urlparse(canon).hostname or canon
    sql += " ORDER BY id DESC LIMIT 500"
    rows = []
    async with _db.execute(sql, args) as cur:
        async for r in cur:
            rows.append(r)

    def _build() -> bytes:
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=15)
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, _pdf_text(f"Audit log — {host}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(110)
        pdf.cell(0, 6, _pdf_text(f"{len(rows)} events — generated {datetime.now():%Y-%m-%d %H:%M}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        for ts, run_id, ev_url, question, answer, grounded, sources, lat, err in rows:
            outcome = "error" if err else ("grounded" if grounded == 1 else "handoff")
            meta = (f"{datetime.fromtimestamp(ts):%Y-%m-%d %H:%M}   {outcome}   "
                    f"{lat / 1000:.1f}s   {run_id}")
            pdf.set_font("helvetica", "", 8)
            pdf.set_text_color(110)
            pdf.multi_cell(0, 5, _pdf_text(meta), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "B", 11)
            pdf.set_text_color(0)
            pdf.multi_cell(0, 6, _pdf_text(f"Q: {question}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(0, 5, _pdf_text(f"A: {err or answer or '(no answer)'}"), new_x="LMARGIN", new_y="NEXT")
            for src in json.loads(sources or "[]"):
                pdf.set_font("helvetica", "", 8)
                pdf.set_text_color(60, 60, 160)
                pdf.multi_cell(0, 4, _pdf_text(f"    source: {src}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0)
            pdf.ln(3)
        return bytes(pdf.output())

    data = await asyncio.to_thread(_build)
    fname = f"audit-{host}-{datetime.now():%Y-%m}.pdf"
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _snapshot(run_id: str, v: dict[str, Any]) -> dict[str, Any]:
    v = v or {}
    return {
        "run_id": run_id,
        "status": v.get("status", "running"),
        "url": v.get("url"),
        "question": v.get("question"),
        "title": _site_title(v),
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
    snap = await _query_graph.aget_state({"configurable": {"thread_id": run_id}})
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
