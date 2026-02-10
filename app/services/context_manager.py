"""
Context window assembly for LLM calls.

Handles: system prompts, RAG retrieval, conversation history,
and automatic summarization of long conversations.
"""

import logging
from dataclasses import dataclass

import tiktoken

from app.config import settings
from app.db.database import db
from app.services.llm_client import llm_client
from app.services.message_cache import message_cache
from app.services.vector_db import vector_db

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    max_tokens: int = settings.default_max_context_tokens
    reserved_for_response: int = settings.reserved_response_tokens
    rag_chunk_budget: int = settings.rag_chunk_token_budget
    summary_threshold: int = settings.summary_threshold_messages


class ContextManager:
    """
    Builds the messages array for each LLM call.

    Token budget priority:
    1. System prompt (always included)
    2. RAG chunks (if agent has a linked collection)
    3. Recent conversation messages (as many as fit)
    4. Conversation summary (if history was truncated)
    5. New user message
    """

    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def build_context(
        self,
        agent_id: str,
        conversation_id: str,
        new_user_message: str,
        config: ContextConfig = None,
    ) -> list[dict]:
        if config is None:
            config = ContextConfig()

        messages = []
        budget = config.max_tokens - config.reserved_for_response

        # ── 1. System prompt ──
        agent = await db.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        system_content = agent["system_prompt"]
        budget -= self.count_tokens(system_content)
        messages.append({"role": "system", "content": system_content})

        # ── 2. RAG retrieval ──
        if agent.get("rag_collection"):
            rag_budget = min(config.rag_chunk_budget, budget // 3)
            rag_chunks = await self._retrieve_rag(
                user_id=str(agent["user_id"]),
                agent_id=str(agent_id),
                query=new_user_message,
                token_budget=rag_budget,
            )
            if rag_chunks:
                rag_text = self._format_rag_chunks(rag_chunks)
                rag_msg = {
                    "role": "system",
                    "content": f"Relevant context from documents:\n\n{rag_text}",
                }
                rag_tokens = self.count_tokens(rag_msg["content"])
                budget -= rag_tokens
                messages.append(rag_msg)

        # ── 3. Conversation history ──
        history = await self._get_history(conversation_id)

        # If conversation is very long, prepend a summary
        if len(history) > config.summary_threshold:
            summary = await self._get_or_create_summary(conversation_id, history)
            if summary:
                summary_msg = {
                    "role": "system",
                    "content": f"Previous conversation summary:\n{summary}",
                }
                summary_tokens = self.count_tokens(summary_msg["content"])
                budget -= summary_tokens
                messages.append(summary_msg)

        # Pack recent messages (newest first to prioritize recency, then reverse)
        recent = []
        for msg in reversed(history):
            msg_tokens = msg.get("token_count") or self.count_tokens(msg["content"])
            if budget - msg_tokens < 0:
                break
            budget -= msg_tokens
            recent.append({"role": msg["role"], "content": msg["content"]})
        recent.reverse()
        messages.extend(recent)

        # ── 4. New user message ──
        messages.append({"role": "user", "content": new_user_message})

        return messages

    async def _retrieve_rag(
        self, user_id: str, agent_id: str, query: str, token_budget: int
    ) -> list[dict]:
        """Get embedding and query vector DB."""
        try:
            embedding_resp = await llm_client.embeddings.create(
                model=settings.embed_model_alias,
                input=query,
            )
            query_vector = embedding_resp.data[0].embedding

            raw_results = vector_db.query(
                user_id=user_id,
                agent_id=agent_id,
                query_vector=query_vector,
                limit=20,
            )

            # Trim to token budget
            chunks = []
            tokens_used = 0
            for result in raw_results:
                chunk_tokens = self.count_tokens(result["text"])
                if tokens_used + chunk_tokens > token_budget:
                    break
                chunks.append(result)
                tokens_used += chunk_tokens

            return chunks
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []

    def _format_rag_chunks(self, chunks: list[dict]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page")
            header = f"[Source: {source}"
            if page is not None:
                header += f", Page {page}"
            header += f", Relevance: {chunk.get('score', 0):.2f}]"
            formatted.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(formatted)

    async def _get_history(self, conversation_id: str) -> list[dict]:
        """Get conversation history from cache or DB."""
        # Try Redis first
        cached = await message_cache.get_recent(conversation_id)
        if cached is not None:
            return cached

        # Cache miss — load from DB and warm cache
        messages = await db.get_messages(conversation_id)
        history = [
            {
                "role": m["role"],
                "content": m["content"],
                "token_count": m.get("token_count"),
            }
            for m in messages
        ]

        if history:
            await message_cache.warm_cache(conversation_id, history)

        return history

    async def _get_or_create_summary(
        self, conversation_id: str, history: list[dict]
    ) -> str:
        """Summarize older messages to compress context."""
        # Check cache
        cached = await message_cache.get_summary(conversation_id)
        if cached:
            return cached

        # Summarize the first 80% of messages
        cutoff = int(len(history) * 0.8)
        old_messages = history[:cutoff]

        text_block = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_messages
        )

        # Truncate if the text block itself is too large
        if self.count_tokens(text_block) > 6000:
            tokens = self.tokenizer.encode(text_block)[:6000]
            text_block = self.tokenizer.decode(tokens)

        try:
            summary_resp = await llm_client.chat.completions.create(
                model=settings.chat_model_alias,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the following conversation concisely. "
                            "Preserve key facts, decisions, user preferences, "
                            "and any important context. Be thorough but brief."
                        ),
                    },
                    {"role": "user", "content": text_block},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            summary = summary_resp.choices[0].message.content
            await message_cache.set_summary(conversation_id, summary)
            return summary
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return ""


context_manager = ContextManager()
