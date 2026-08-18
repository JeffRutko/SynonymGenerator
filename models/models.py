import os
from urllib.parse import quote_plus


MCP_SERVER_URLS = [
    "https://kenzo19125-search-tools-mcp.hf.space/mcp",
    "https://kenzo19125-cohere-tools-mcp.hf.space/mcp",
]

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"


def _normalize_database_url(raw: str) -> str:
    url = raw.strip().strip('"').strip("'")
    while url.upper().startswith("DATABASE_URL="):
        url = url.split("=", 1)[1].strip()
    if url.startswith("jdbc:"):
        url = url[5:]
    return url


def _is_placeholder_url(url: str) -> bool:
    lower = url.lower()
    return (
        "user:password" in lower
        or "<password>" in lower
        or "<user>" in lower
        or "your_password" in lower
    )


def _build_database_url_from_pg_env() -> str:
    user = os.environ.get("PGUSER", "").strip()
    password = os.environ.get("PGPASSWORD", "").strip()
    host = os.environ.get("PGHOST", "").strip()
    port = os.environ.get("PGPORT", "5432").strip() or "5432"
    database = os.environ.get("PGDATABASE", "neondb").strip() or "neondb"
    if not all([user, password, host]):
        return ""
    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{database}?sslmode=require"
    )


def _resolve_database_url() -> str:
    direct = _normalize_database_url(os.environ.get("DATABASE_URL", ""))
    if direct and not _is_placeholder_url(direct):
        return direct
    return _build_database_url_from_pg_env()


DATABASE_URL = _resolve_database_url()
DB_ENABLED = bool(DATABASE_URL)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DB_CACHE_TTL_HOURS = _env_int("DB_CACHE_TTL_HOURS", 168)
DB_VECTOR_DIMENSIONS = _env_int("DB_VECTOR_DIMENSIONS", 1024)


def create_hf_model():
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        client_args={
            "api_key": os.environ["HF_TOKEN"],
            "base_url": HF_BASE_URL,
        },
        model_id=HF_MODEL_ID,
    )