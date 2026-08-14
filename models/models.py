import os
from strands.models.openai import OpenAIModel


MCP_SERVER_URLS = [
    "https://kenzo19125-search-tools-mcp.hf.space/mcp",
    "https://kenzo19125-cohere-tools-mcp.hf.space/mcp",
]

HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"

MONGODB_URI = os.environ.get("MONGODB_URI", "").strip()
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "synonym_generator")
MONGODB_ENABLED = bool(MONGODB_URI)
MONGODB_CACHE_TTL_HOURS = int(os.environ.get("MONGODB_CACHE_TTL_HOURS", "168"))
MONGODB_VECTOR_INDEX_NAME = os.environ.get(
    "MONGODB_VECTOR_INDEX_NAME", "chunk_vectors_vector_index"
)
MONGODB_VECTOR_DIMENSIONS = int(os.environ.get("MONGODB_VECTOR_DIMENSIONS", "1024"))


def create_hf_model() -> OpenAIModel:
    return OpenAIModel(
        client_args={
            "api_key": os.environ["HF_TOKEN"],
            "base_url": HF_BASE_URL,
        },
        model_id=HF_MODEL_ID,
    )