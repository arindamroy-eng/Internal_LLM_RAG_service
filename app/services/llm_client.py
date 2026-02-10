"""OpenAI-compatible client that routes through LiteLLM to internal vLLM instances."""

from openai import AsyncOpenAI
from app.config import settings


def get_llm_client() -> AsyncOpenAI:
    """Get an async OpenAI client pointed at the LiteLLM proxy."""
    return AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_master_key,
    )


# Singleton client
llm_client = get_llm_client()
