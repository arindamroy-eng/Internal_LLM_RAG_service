"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── App ──
    app_name: str = "Internal_LLM_RAG_service"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    debug: bool = False
    log_level: str = "info"

    # ── Auth ──
    jwt_secret_key: str = "change-me-to-a-random-64-char-string"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    admin_api_key: str = "sk-admin-change-me"

    # ── PostgreSQL ──
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "llm_platform"
    postgres_user: str = "llm_admin"
    postgres_password: str = "change-me"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_message_ttl: int = 86400
    # Must match redis_message_ttl. When the summary expired earlier than the
    # messages it was derived from, the next turn regenerated DIFFERENT
    # summary text and invalidated the prompt prefix from position two,
    # discarding the whole conversation's prefix cache.
    redis_summary_ttl: int = 86400

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # ── Qdrant ──
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None
    embedding_dimension: int = 1024  # BGE-large

    # ── LiteLLM ──
    litellm_host: str = "localhost"
    litellm_port: int = 4000
    litellm_master_key: str = "sk-litellm-master-change-me"

    @property
    def litellm_base_url(self) -> str:
        return f"http://{self.litellm_host}:{self.litellm_port}/v1"

    # ── LLM HTTP client tuning ──
    # Retries live in LiteLLM (num_retries), not in the SDK — see
    # app/services/llm_client.py for why stacking them is dangerous.
    llm_request_timeout: float = 600.0
    llm_connect_timeout: float = 5.0
    llm_max_connections: int = 1000
    llm_max_keepalive_connections: int = 200

    # ── Model Aliases ──
    # These are OpenAI-shaped names for SELF-HOSTED models; no OpenAI model is
    # ever called. The alias tells you nothing about the backend:
    #   gpt-4 / gpt-4-turbo / gpt-4o -> Llama 3.1 405B Instruct (MXFP4)
    #   gpt-3.5-turbo                -> Qwen 2.5 Coder 32B  (a CODE model)
    #   text-embedding-*             -> BGE-large-en-v1.5
    chat_model_alias: str = "gpt-4"
    code_model_alias: str = "gpt-3.5-turbo"
    embed_model_alias: str = "text-embedding-ada-002"

    # Chat-capable aliases beyond chat_model_alias and code_model_alias.
    # Embedding aliases are deliberately excluded — an agent cannot converse
    # with an embedding model.
    #
    # MUST stay in sync with the model_list in litellm_config.yaml. A value
    # accepted here but absent there is accepted at agent-creation time and
    # then fails as a 502 on the user's first chat.
    extra_agent_model_aliases: str = "gpt-4-turbo,gpt-4o"

    @property
    def allowed_agent_models(self) -> tuple[str, ...]:
        """Aliases accepted for `agents.model`, in display order."""
        extras = [
            a.strip() for a in self.extra_agent_model_aliases.split(",") if a.strip()
        ]
        # dict.fromkeys dedupes while preserving order.
        return tuple(
            dict.fromkeys([self.chat_model_alias, self.code_model_alias, *extras])
        )

    # ── Context Management ──
    default_max_context_tokens: int = 16384
    reserved_response_tokens: int = 4096
    rag_chunk_token_budget: int = 3000
    summary_threshold_messages: int = 50
    # History window eviction. The window is anchored and evicted in blocks so
    # the prompt prefix stays byte-identical across consecutive turns; see
    # ContextManager._select_history_window. Larger blocks trade context
    # density for prefix stability (and therefore prefix-cache hit rate).
    history_evict_block_messages: int = 8
    history_evict_target_ratio: float = 0.85

    # ── Document Ingestion ──
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_batch_size: int = 64
    max_upload_size_mb: int = 100
    allowed_extensions: str = "pdf,docx,txt,md,csv,html"

    @property
    def allowed_extensions_set(self) -> set:
        return set(self.allowed_extensions.split(","))

    # ── Rate Limits ──
    default_rpm: int = 60
    default_tpm: int = 200000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
