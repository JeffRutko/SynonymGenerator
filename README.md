# Conceptual Search Helper

AI assistant for telecom and patent prior-art research. Enter a concept (and an optional CPC / domain hint) and get:

- Synonym families and related terminology
- **CPC subgroups** (not just section-level codes)
- Boolean search-string variants
- Brief notes on why terms and codes fit

Built with **FastAPI**, a **Strands** agent, **MCP** search/embed tools, and optional **Neon PostgreSQL** RAG caching. Progress and markdown results stream over **Server-Sent Events**.

## How it works

```
User → Web UI → FastAPI (SSE)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Cache check           Full pipeline
 (syn_synonym_outputs)          │
         │                     ▼
    hit → stream        Gather MCP sources
                        (telecom / patent / web)
                               │
                               ▼
                         Chunk passages
                               │
                               ▼
                    Embed + Vector Search
                     (Cohere + pgvector)
                               │
                               ▼
                      Strands agent
                   (DeepSeek via HF)
                   + optional tool calls
                               │
                               ▼
                    Cache report → SSE UI
```

Without `DATABASE_URL`, the same pipeline runs with an ephemeral in-memory Chroma store (no persistent cache).

## Quick start

```bash
uv sync
cp .env.example .env
# set HF_TOKEN (and optionally DATABASE_URL) in .env
uv run python app.py
```

Open **http://127.0.0.1:7860** (use `localhost` or `127.0.0.1`, not `0.0.0.0`).

Defaults: `HOST=127.0.0.1`, `PORT=7860`. For containers, set `HOST=0.0.0.0`.

### Social / LinkedIn preview

The homepage includes Open Graph tags. After deploy, the share image is:

`https://YOUR-APP.up.railway.app/static/og-image.png`

LinkedIn **Projects** usually need a manual media upload — download that PNG and attach it to the project. For link-share previews, refresh the scrape with [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/).

### Environment variables

| Name | Required | Notes |
|------|----------|--------|
| `HF_TOKEN` | Yes | Hugging Face token for the Inference Router / model in `models/models.py` |
| `DATABASE_URL` | No | Neon `postgresql://…` (pooler URL recommended); when unset, RAG uses ephemeral Chroma only |
| `DB_CACHE_TTL_HOURS` | No | Soft cache window for app reads; default `168` (7 days) |
| `DB_VECTOR_DIMENSIONS` | No | Default `1024` (must match Cohere embed + pgvector column) |

MCP server URLs (search + Cohere embed/rerank) and the model ID live in [`models/models.py`](models/models.py).

**Do not commit secrets.** Keep credentials in `.env` (gitignored) or your host’s secret store. See [`.env.example`](.env.example).

## Usage

1. Enter a concept or phrase (for example, *the base station transmits a packet to a UE*).
2. Optionally add a domain or CPC hint (for example, `H04W`).
3. Click **Generate search help**.

Streaming progress and markdown arrive from `POST /v1/search` (SSE).

Optional JSON field `"force_refresh": true` bypasses database caches for that request.

### Smoke checks

```bash
# Neon PostgreSQL connection (+ creates syn_* tables on first connect)
uv run python scripts/test_db_connection.py

# pgvector search on syn_chunk_vectors
uv run python scripts/test_vector_search.py

# API health
curl -s http://127.0.0.1:7860/health

# CLI agent path
uv run python agent.py
```

## PostgreSQL caching (optional)

When `DATABASE_URL` is set, repeat searches reuse:

| Table | Contents |
|-------|----------|
| `syn_source_documents` | Cached MCP source text per tool |
| `syn_chunk_vectors` | Chunk text + embeddings (pgvector HNSW) |
| `syn_synonym_outputs` | Final markdown reports |

Cache keys are derived from concept + context. Soft expiry uses `expires_at` / `DB_CACHE_TTL_HOURS`. Tables are created automatically on startup via [`db/migrations/001_syn_tables.sql`](db/migrations/001_syn_tables.sql).

## Deploy (Railway)

The repo includes [`railway.toml`](railway.toml), [`nixpacks.toml`](nixpacks.toml), and [`Procfile`](Procfile). Nixpacks installs from [`requirements.txt`](requirements.txt) (not `uv.lock`) and starts:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

1. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select this repository (`master`)
3. **Settings → Networking** → **Generate Domain**
4. Set variables (below). Do **not** set `HOST` or `PORT` — the start command already uses `$PORT`.

| Name | Required | Notes |
|------|----------|-------|
| `HF_TOKEN` | Yes | Same token used locally |
| `DATABASE_URL` | Yes for persistent cache | Neon connection string (use pooler endpoint) |
| `DB_CACHE_TTL_HOURS` | No | Default `168` |
| `DB_VECTOR_DIMENSIONS` | No | Default `1024` |

### Verify

```bash
curl -s https://YOUR-APP.up.railway.app/health
# {"status":"ok","db":"connected"}

uv run python scripts/check_deploy_health.py https://YOUR-APP.up.railway.app
```

| `/health` `db` value | Meaning |
|----------------------|---------|
| `connected` | OK |
| `disabled` | `DATABASE_URL` not set |
| `error` | Connection failure — check Railway logs and Neon credentials |

MCP servers and the HF model need outbound HTTPS from Railway.

## Stack

- **API / UI:** FastAPI, SSE, static frontend
- **Agent:** Strands + Hugging Face Inference Router (DeepSeek)
- **Tools:** MCP — `telecom_search`, `patent_search`, `web_text_search`, Cohere `embed_texts` / `rerank_documents`
- **RAG:** chunking, embeddings, pgvector (or ephemeral Chroma)
- **Data:** Neon PostgreSQL (optional multi-layer cache in `syn_*` tables)
- **Deploy:** Railway + Nixpacks

## License

This project is licensed under the [MIT License](LICENSE).
