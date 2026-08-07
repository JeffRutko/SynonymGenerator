# Conceptual Search Helper

FastAPI app that uses a Strands agent with MCP search tools
(`telecom_search`, `patent_search`, and related tools) plus ephemeral Chroma RAG
to propose synonyms, **CPC subgroups**, and Boolean search strings for telecom /
patent concepts.

## Usage

1. Enter a concept or phrase (for example, *the base station transmits a packet to a UE*).
2. Optionally add a domain or CPC hint (for example, `H04W`).
3. Click **Generate search help**.

Streaming progress and markdown results arrive over `POST /v1/search` (SSE).

## Local development

```bash
uv sync
# put HF_TOKEN in a local .env
uv run python app.py
```

Open **http://127.0.0.1:7860** (or `localhost`). Do not use `http://0.0.0.0:7860` — browsers reject that address.

Defaults: `HOST=127.0.0.1`, `PORT=7860`. For container deploys set `HOST=0.0.0.0`.

| Name | Required | Notes |
|------|----------|--------|
| `HF_TOKEN` | Yes | Hugging Face Inference Router / model used in `models/models.py` |

The MCP search server URL and model ID live in `models/models.py`.

Each request gathers MCP search hits, chunks them (~500 chars, **15% overlap**),
embeds with MiniLM via HF Inference, retrieves top passages from an **ephemeral**
Chroma collection, then runs the Strands agent grounded on that context.

CLI smoke test:

```bash
uv run python agent.py
```

API smoke check:

```bash
curl -s http://127.0.0.1:7860/health
```

## Deploy

Railway hosting is planned next (separate from this local FastAPI + UI).
