# Conceptual Search Helper

FastAPI app that uses a Strands agent with MCP search tools
(`telecom_search`, `patent_search`, and related tools) plus RAG (MongoDB Atlas
when configured, otherwise ephemeral Chroma) to propose synonyms, **CPC
subgroups**, and Boolean search strings for telecom / patent concepts.

## Usage

1. Enter a concept or phrase (for example, *the base station transmits a packet to a UE*).
2. Optionally add a domain or CPC hint (for example, `H04W`).
3. Click **Generate search help**.

Streaming progress and markdown results arrive over `POST /v1/search` (SSE).

Optional JSON field `"force_refresh": true` bypasses MongoDB caches for that request.

## Local development

```bash
uv sync
# put HF_TOKEN and MONGODB_URI in a local .env
uv run python app.py
```

MongoDB Atlas connection test:

```bash
uv run python scripts/test_mongo_connection.py
```

Atlas Vector Search index test:

```bash
uv run python scripts/test_vector_search.py
```

Open **http://127.0.0.1:7860** (or `localhost`). Do not use `http://0.0.0.0:7860` — browsers reject that address.

Defaults: `HOST=127.0.0.1`, `PORT=7860`. For container deploys set `HOST=0.0.0.0`.

| Name | Required | Notes |
|------|----------|--------|
| `HF_TOKEN` | Yes | Hugging Face Inference Router / model used in `models/models.py` |
| `MONGODB_URI` | No | Atlas `mongodb+srv://…`; when unset, RAG uses ephemeral Chroma only |
| `MONGODB_DB_NAME` | No | Default `synonym_generator` |
| `MONGODB_CACHE_TTL_HOURS` | No | Default `168` (7 days) for cached sources/outputs |
| `MONGODB_VECTOR_INDEX_NAME` | No | Default `chunk_vectors_vector_index` |
| `MONGODB_VECTOR_DIMENSIONS` | No | Default `1024` (must match Cohere embed + Atlas index) |

MCP server URLs (search + Cohere embed/rerank) and the model ID live in
`models/models.py`.

With MongoDB configured, repeat searches cache MCP source text, chunk
embeddings, and final reports under the same concept + context. Without
MongoDB, each request uses in-memory Chroma only.

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

If `$vectorSearch` is unavailable, the app falls back to in-process cosine
similarity over cached embeddings.

CLI smoke test:

```bash
uv run python agent.py
```

API smoke check:

```bash
curl -s http://127.0.0.1:7860/health
```

## Deploy (Railway)

The repo includes [`railway.toml`](railway.toml), [`nixpacks.toml`](nixpacks.toml),
and [`Procfile`](Procfile). Nixpacks installs from [`requirements.txt`](requirements.txt)
(not `uv.lock`) and starts:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 1. Connect GitHub

1. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo**
2. Select `JeffRutko/SynonymGenerator` (branch `master`)
3. **Settings → Networking** → **Generate Domain** (e.g. `your-app.up.railway.app`)

No Railway MongoDB plugin — use Atlas via `MONGODB_URI`.

### 2. Environment variables

Railway → service → **Variables**:

| Name | Required | Notes |
|------|----------|-------|
| `HF_TOKEN` | Yes | Same token used locally |
| `MONGODB_URI` | Yes | Atlas `mongodb+srv://…` connection string |
| `MONGODB_DB_NAME` | No | Default `synonym_generator` |
| `MONGODB_CACHE_TTL_HOURS` | No | Default `168` |
| `MONGODB_VECTOR_INDEX_NAME` | No | Default `chunk_vectors_vector_index` |
| `MONGODB_VECTOR_DIMENSIONS` | No | Default `1024` |

Do **not** set `HOST` or `PORT` — the start command already uses `$PORT`.

### 3. Atlas (one-time)

- **Network Access:** allow Railway egress (`0.0.0.0/0` or Railway static IP)
- **Vector Search index** on `chunk_vectors` — see [Atlas Vector Search index](#atlas-vector-search-index) above
- DB user in the URI needs read/write on `synonym_generator`

### 4. Verify deploy

```bash
curl -s https://YOUR-APP.up.railway.app/health
# {"status":"ok","mongo":"connected"}

uv run python scripts/check_deploy_health.py https://YOUR-APP.up.railway.app
```

Open `https://YOUR-APP.up.railway.app/`, run a search, then repeat the same query to
confirm MongoDB caching (second run should be much faster).

### Troubleshooting

| `/health` mongo value | Cause |
|-----------------------|-------|
| `connected` | OK |
| `disabled` | `MONGODB_URI` not set on Railway |
| `error` | Atlas network/auth failure — check Railway logs and Atlas IP allowlist |

MCP servers (search + Cohere embed) and the HF model ID are configured in
[`models/models.py`](models/models.py); they require outbound HTTPS from Railway.
