---
title: Conceptual Search Helper
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
---

# Conceptual Search Helper

Single-shot Gradio app that uses a Strands agent with MCP search tools
(`telecom_search`, `patent_search`, and related tools) plus ephemeral Chroma RAG
to propose synonyms, **CPC subgroups**, and Boolean search strings for telecom /
patent concepts.

## Usage

1. Enter a concept or phrase (for example, *the base station transmits a packet to a UE*).
2. Optionally add a domain or CPC hint (for example, `H04W`).
3. Click **Generate search help**.

## Hugging Face Space setup

Add a Space **secret**:

| Name | Required | Notes |
|------|----------|--------|
| `HF_TOKEN` | Yes | Token with access to the Hugging Face Inference Router / model used in `models/models.py` |

The MCP search server URL and model ID live in `models/models.py`.

## Local development

```bash
uv sync
# put HF_TOKEN in a local .env
uv run python app.py
```

Each request gathers MCP search hits, chunks them (~500 chars, **15% overlap**),
embeds with MiniLM via HF Inference, retrieves top passages from an **ephemeral**
Chroma collection, then runs the Strands agent grounded on that context.

CLI smoke test:

```bash
uv run python agent.py
```
