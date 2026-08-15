# Conceptual Search Helper

AI assistant for telecom and patent prior-art research. Enter a concept (and an optional CPC / domain hint) and get:

- Synonym families and related terminology
- **CPC subgroups** (not just section-level codes)
- Boolean search-string variants
- Brief notes on why terms and codes fit

Built with **FastAPI**, a **Strands** agent, **MCP** search/embed tools, and optional **MongoDB Atlas** RAG caching. Progress and markdown results stream over **Server-Sent Events**.

## How it works

```
User → Web UI → FastAPI (SSE)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Cache check           Full pipeline
 (synonym_outputs)             │
         │                     ▼
    hit → stream        Gather MCP sources
                        (telecom / patent / web)
                               │
                               ▼
                         Chunk passages
                               │
                               ▼
                    Embed + Vector Search
                     (Cohere + Atlas)
                               │
                               ▼
                      Strands agent
                   (DeepSeek via HF)
                   + optional tool calls
                               │
                               ▼
                    Cache report → SSE UI
```

Without `MONGODB_URI`, the same pipeline runs with an ephemeral in-memory Chroma store (no persistent cache).

## Quick start

```bash
uv sync
cp .env.example .env
# set HF_TOKEN (and optionally MONGODB_URI) in .env
uv run python app.py
```

Open **http://127.0.0.1:7860** (use `localhost` or `127.0.0.1`, not `0.0.0.0`).

Defaults: `HOST=127.0.0.1`, `PORT=7860`. For containers, set `HOST=0.0.0.0`.

### Environment variables

| Name | Required | Notes |
|------|----------|--------|
| `HF_TOKEN` | Yes | Hugging Face token for the Inference Router / model in `models/models.py` |
| `MONGODB_URI` | No | Atlas `mongodb+srv://…`; when unset, RAG uses ephemeral Chroma only |
| `MONGODB_DB_NAME` | No | Default `synonym_generator` |
| `MONGODB_CACHE_TTL_HOURS` | No | Soft cache window for app reads; default `168` (7 days) |
| `MONGODB_VECTOR_INDEX_NAME` | No | Default `chunk_vectors_vector_index` |
| `MONGODB_VECTOR_DIMENSIONS` | No | Default `1024` (must match Cohere embed + Atlas index) |

MCP server URLs (search + Cohere embed/rerank) and the model ID live in [`models/models.py`](models/models.py).

**Do not commit secrets.** Keep credentials in `.env` (gitignored) or your host’s secret store. See [`.env.example`](.env.example).

## Usage

1. Enter a concept or phrase (for example, *the base station transmits a packet to a UE*).
2. Optionally add a domain or CPC hint (for example, `H04W`).
3. Click **Generate search help**.

Streaming progress and markdown arrive from `POST /v1/search` (SSE).

Optional JSON field `"force_refresh": true` bypasses MongoDB caches for that request.

### Smoke checks

```bash
# MongoDB Atlas connection
uv run python scripts/test_mongo_connection.py

# Atlas Vector Search index
uv run python scripts/test_vector_search.py

# API health
curl -s http://127.0.0.1:7860/health

# CLI agent path
uv run python agent.py
```

## MongoDB caching (optional)

When `MONGODB_URI` is set, repeat searches reuse:

| Collection | Contents |
|------------|----------|
| `source_documents` | Cached MCP source text per tool |
| `chunk_vectors` | Chunk text + embeddings (Vector Search) |
| `synonym_outputs` | Final markdown reports |

Cache keys are derived from concept + context. Soft expiry uses `expires_at` / `MONGODB_CACHE_TTL_HOURS`. For physical deletion in Atlas, configure a TTL index separately (for example on `created_at`).

### Atlas Vector Search index

Create on collection `chunk_vectors` (Atlas Search → JSON Editor):

```json
{
  "name": "chunk_vectors_vector_index",
  "type": "vectorSearch",
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "query_key"
    }
  ]
}
```

If `$vectorSearch` is unavailable, the app falls back to in-process cosine similarity over cached embeddings.

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
| `MONGODB_URI` | Yes for persistent cache | Atlas connection string |
| `MONGODB_DB_NAME` | No | Default `synonym_generator` |
| `MONGODB_CACHE_TTL_HOURS` | No | Default `168` |
| `MONGODB_VECTOR_INDEX_NAME` | No | Default `chunk_vectors_vector_index` |
| `MONGODB_VECTOR_DIMENSIONS` | No | Default `1024` |

### Atlas (one-time)

- **Network Access:** allow Railway egress (`0.0.0.0/0` or Railway static IPs)
- **Vector Search index** on `chunk_vectors` (see above)
- DB user in the URI needs read/write on the database

### Verify

```bash
curl -s https://YOUR-APP.up.railway.app/health
# {"status":"ok","mongo":"connected"}

uv run python scripts/check_deploy_health.py https://YOUR-APP.up.railway.app
```

| `/health` `mongo` value | Meaning |
|-------------------------|---------|
| `connected` | OK |
| `disabled` | `MONGODB_URI` not set |
| `error` | Atlas network/auth failure — check Railway logs and Atlas IP allowlist |

**`No replica set members found yet` / `ReplicaSetNoPrimary`:** almost always Atlas Network Access. Allow `0.0.0.0/0` (or Railway egress IPs), wait a minute, then retry `/health`. Confirm `MONGODB_URI` is a full `mongodb+srv://…` string and that special characters in the password are URL-encoded.

MCP servers and the HF model need outbound HTTPS from Railway.

## Stack

- **API / UI:** FastAPI, SSE, static frontend
- **Agent:** Strands + Hugging Face Inference Router (DeepSeek)
- **Tools:** MCP — `telecom_search`, `patent_search`, `web_text_search`, Cohere `embed_texts` / `rerank_documents`
- **RAG:** chunking, embeddings, Atlas Vector Search (or ephemeral Chroma)
- **Data:** MongoDB Atlas (optional multi-layer cache)
- **Deploy:** Railway + Nixpacks

## License

This project is licensed under the [MIT License](LICENSE).
