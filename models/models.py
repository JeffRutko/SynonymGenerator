import os
from strands.models.openai import OpenAIModel


MCP_SERVER_URL = "https://kenzo19125-search-tools-mcp.hf.space/mcp"
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"


def create_hf_model() -> OpenAIModel:
    return OpenAIModel(
        client_args={
            "api_key": os.environ["HF_TOKEN"],
            "base_url": HF_BASE_URL,
        },
        model_id=HF_MODEL_ID,
    )