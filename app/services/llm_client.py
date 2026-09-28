"""OpenAI-compatible client that routes through LiteLLM to internal vLLM instances."""

import httpx
from openai import AsyncOpenAI

from app.config import settings


def get_llm_client() -> AsyncOpenAI:
    """Get an async OpenAI client pointed at the LiteLLM proxy."""
    return AsyncOpenAI(
        base_url=settings.litellm_base_url,
        api_key=settings.litellm_master_key,
        # Retries are owned by LiteLLM (num_retries in litellm_config.yaml),
        # NOT here. The SDK defaults to max_retries=2, which multiplies with
        # LiteLLM's own retries: a single user request could become 12
        # attempts at a 30k-token prefill. Each retry also re-enters the
        # endpoint picker and may land on a different replica, throwing away
        # the prefix cache. One layer of retries only.
        max_retries=0,
        timeout=httpx.Timeout(
            settings.llm_request_timeout,
            connect=settings.llm_connect_timeout,
        ),
        http_client=httpx.AsyncClient(
            # The SDK default pool is 100 connections / 20 keepalive, which
            # saturates well below this platform's concurrency target.
            limits=httpx.Limits(
                max_connections=settings.llm_max_connections,
                max_keepalive_connections=settings.llm_max_keepalive_connections,
            ),
            timeout=httpx.Timeout(
                settings.llm_request_timeout,
                connect=settings.llm_connect_timeout,
            ),
        ),
    )


# Singleton client
llm_client = get_llm_client()
