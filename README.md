# hustle-stack

Self-contained dev/demo stack in one `docker compose up`: **Ollama** (local
LLM + embeddings), **Playwright** (shared headless-browser server), and
**Firecrawl** (self-hosted web-scraping API, wired to use the project's
Ollama for its LLM features).

Everything runs from published images — no host installs of ollama, node,
playwright, or firecrawl are used. To move to a VPS, copy this repo, set
`.env`, and run the same command. All state lives in named volumes.

## Quick start

```bash
cp .env.example .env        # optional — defaults work out of the box
docker compose up -d
```

First boot pulls images (~5 GB) and then the ollama models
(~2.5 GB: `llama3.2:3b` + `nomic-embed-text`). Watch progress with
`docker compose logs -f ollama-pull`.

## Endpoints

| Service    | Local URL                          | Notes                          |
| ---------- | ---------------------------------- | ------------------------------ |
| Ollama     | `http://localhost:11435`           | OpenAI-compat at `/v1`         |
| Firecrawl  | `http://localhost:3002`            | `POST /v2/scrape`, `/v2/crawl` |
| Playwright | `ws://localhost:3100/browser`      | *opt-in*, see below            |

> **Do I need the standalone `playwright` service?** Not for scraping —
> Firecrawl bundles its own internal render service (`firecrawl-playwright`)
> and uses it automatically. The standalone service only matters if you want
> to drive a browser from *your own* code (`chromium.connect`) for things
> Firecrawl can't do: custom automation, screenshots, logged-in flows. It's
> opt-in so it doesn't eat ~2 GB of RAM on every `up`:
>
> ```bash
> docker compose --profile browser up -d playwright
> ```

> **Port note:** a host-installed Ollama typically owns `localhost:11434`.
> This project's container publishes on **11435** so the two never conflict.
> Inside the compose network, services always reach it at
> `http://ollama:11434` regardless of the published port. If you deploy to a
> VPS with no host ollama, set `OLLAMA_PORT=11434` in `.env`.

## Smoke tests

```bash
# Ollama is serving + models are pulled
curl http://localhost:11435/api/tags

# Playwright browser server is listening (ws endpoint in logs)
docker compose logs playwright | grep "ws://"

# Firecrawl scrape → markdown
curl -s -X POST http://localhost:3002/v2/scrape \
  -H "content-type: application/json" \
  -d '{"url":"https://example.com","formats":["markdown"]}'

# Full containerized test suite (no host tooling needed)
docker compose --profile test run --rm e2e
```

## Architecture

```
┌─────────────┐   ws://playwright:3000/browser   (opt-in `browser` profile)
│ playwright  │ ◄────── your scripts/tests connect here
│ (mcr image, │        (chromium.connect) — not needed for scraping
│  chromium)  │
└─────────────┘
┌─────────────┐
│ ollama      │ ◄────── http://ollama:11434 (+ OpenAI-compat /v1)
│ llama3.2 +  │
│ nomic-embed │
└──────┬──────┘
       │ OLLAMA_BASE_URL (firecrawl's LLM features point here)
┌──────┴───────────┐
│ firecrawl-api    │ :3002 — REST API (harness runs api+workers)
│  ├─ firecrawl-   │      page-render microservice (its own browsers,
│  │  playwright   │      NOT the standalone service above)
│  ├─ firecrawl-   │      redis — BullMQ/rate limiting
│  │  redis        │
│  ├─ firecrawl-   │      rabbitmq — Nuq job queue
│  │  rabbitmq     │
│  └─ firecrawl-   │      nuq postgres — job/crawl state
│     postgres     │
└──────────────────┘
```

Two "playwright" things exist: `playwright` (opt-in `browser`/`test`
profile) is a generic browser-server endpoint for your own tooling;
`firecrawl-playwright` is Firecrawl's internal render backend — required
for JS-rendered scraping, runs always, and is not published to the host.

## Usage snippets

Remote browser from your own scripts:

```ts
import { chromium } from "playwright";
const browser = await chromium.connect("ws://localhost:3100/browser");
```

Ollama chat (OpenAI-compatible):

```bash
curl http://localhost:11435/v1/chat/completions -d '{
  "model":"llama3.2:3b",
  "messages":[{"role":"user","content":"hello"}]}'
```

Firecrawl scrape:

```bash
curl -X POST http://localhost:3002/v2/scrape \
  -H "content-type: application/json" \
  -d '{"url":"https://example.com","formats":["markdown"]}'
```

## Environment

See `.env.example` — model names, published ports, firecrawl concurrency.
GPU: the `ollama` service has a commented `deploy` block for NVIDIA
reservations (needs `nvidia-container-toolkit` on the host).

## Production / VPS notes

- All published ports bind to `127.0.0.1` — put a reverse proxy in front
  and keep it that way, or firewall the ports.
- `USE_DB_AUTHENTICATION=false` means **Firecrawl has no API key** — fine on
  the compose bridge/localhost, not acceptable exposed to the internet.
- The standalone playwright server has **no auth** — anyone who can reach
  the port controls a browser. Keep it localhost/firewalled, or drop the
  port mapping entirely on the VPS (clients inside the network can still
  use `ws://playwright:3000/browser`).
- Set a real `FC_POSTGRES_PASSWORD` before deploying.
- Backup: `docker run --rm -v hustle-stack_ollama-models:/v alpine tar cf - /v`
  (same pattern for `fc-pg-data`, `fc-redis-data`).

## Limits

- `ollama-pull` re-checks models on every `up` (idempotent; seconds when
  already pulled). Change models in `.env`.
- Firecrawl self-host lacks the cloud "fire-engine" — no managed
  anti-bot/proxy layer. Respect target sites' terms.
- Image tags `latest` are used for ollama/firecrawl for dev convenience;
  pin digests for a reproducible VPS deployment.
