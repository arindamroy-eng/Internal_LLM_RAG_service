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
    # Number of messages dropped at once when the history window overflows.
    # Larger blocks mean the prompt prefix stays byte-identical for more
    # consecutive turns; the cost is discarding still-affordable history.
    evict_block_messages: int = settings.history_evict_block_messages
    # After an eviction, keep evicting down to this fraction of the budget so
    # the very next turn does not immediately evict again (hysteresis).
    evict_target_ratio: float = settings.history_evict_target_ratio


class ContextManager:
    """
    Builds the messages array for each LLM call.

    Message order is chosen so that the longest possible PREFIX is stable
    across consecutive turns of the same conversation:

        1. System prompt          — fixed per agent
        2. Conversation summary   — changes rarely
        3. Conversation history   — anchored window, grows at the tail
        4. RAG chunks             — re-retrieved every turn, so it goes LAST
        5. New user message       — volatile by definition

    The RAG block used to sit at position 2. Because its content changes with
    every query, every token after it was a prefix-cache miss on every turn,
    leaving the system prompt as the only reusable span (~3% of a 16k
    context). Moving it after the history makes roughly 70-85% of the prompt
    reusable.

    This matters beyond raw vLLM efficiency: the llm-d endpoint picker routes
    requests to the replica that already holds their prefix, so with no stable
    prefix there is nothing to route on.

    Two invariants keep that prefix stable, and both are load-bearing:

      * The history budget is reduced by a FIXED rag_reserve, never by the
        actual retrieved size. Otherwise a 2400-token retrieval and a
        2900-token retrieval admit different numbers of history messages and
        the history prefix shifts anyway.
      * The history window is anchored on a persisted monotonic message seq
        and evicts in blocks, never one message at a time.
    """

    def __init__(self):
        # NOTE: cl100k_base is GPT-4's BPE, not Llama 3.1's or Qwen's. Token
        # counts here are approximations used for budgeting only. They are
        # deliberately consistent rather than exact — a stable overestimate
        # keeps the window anchor from oscillating.
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

        # ── 1. System prompt (stable per agent) ──
        agent = await db.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        system_content = agent["system_prompt"]
        budget -= self.count_tokens(system_content)
        messages.append({"role": "system", "content": system_content})

        # ── Reserve the RAG allowance up front, at a FIXED size ──
        # Retrieval happens later, but its budget is subtracted now and is not
        # refunded if the retrieved chunks come in under it. Sizing the
        # history window off the ACTUAL retrieved size would make the window
        # (and therefore the prompt prefix) vary with each query's retrieval,
        # which is exactly the instability this ordering exists to remove.
        rag_enabled = bool(agent.get("rag_collection"))
        rag_reserve = config.rag_chunk_budget if rag_enabled else 0
        budget -= rag_reserve

        # ── 2. Conversation summary (changes rarely) ──
        history = await self._get_history(conversation_id)

        if len(history) > config.summary_threshold:
            summary = await self._get_or_create_summary(conversation_id, history)
            if summary:
                summary_msg = {
                    "role": "system",
                    "content": f"Previous conversation summary:\n{summary}",
                }
                budget -= self.count_tokens(summary_msg["content"])
                messages.append(summary_msg)

        # ── 3. Conversation history (anchored window) ──
        # Reserve room for the new user message so it cannot push the window.
        budget -= self.count_tokens(new_user_message)
        retained = await self._select_history_window(
            conversation_id=conversation_id,
            history=history,
            budget=budget,
            config=config,
        )
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in retained
        )

        # ── 4. RAG chunks (volatile — deliberately after the history) ──
        if rag_enabled:
            rag_chunks = await self._retrieve_rag(
                user_id=str(agent["user_id"]),
                agent_id=str(agent_id),
                query=new_user_message,
                token_budget=rag_reserve,
            )
            if rag_chunks:
                messages.append(
                    {
                        "role": "system",
                        "content": self._format_rag_block(rag_chunks),
                    }
                )

        # ── 5. New user message ──
        messages.append({"role": "user", "content": new_user_message})

        return messages

    async def _select_history_window(
        self,
        conversation_id: str,
        history: list[dict],
        budget: int,
        config: ContextConfig,
    ) -> list[dict]:
        """
        Choose which history messages to include, using a persisted anchor.

        The naive approach — drop the oldest message until the remainder fits
        — moves the window by one message on every turn, so the serialized
        prefix changes every turn and the prefix cache is never reused.

        Instead we keep a monotonic `window_start` (a message `seq`) in Redis
        and only ever advance it, in blocks, with hysteresis. The window is
        then stable for several consecutive turns and jumps once, so reuse
        over the history span is roughly 1 - (evict_block / turns).
        """
        if not history:
            return []

        anchor = await message_cache.get_window_start(conversation_id) or 0

        def _fits(msgs: list[dict]) -> bool:
            return self._history_tokens(msgs) <= budget

        retained = [m for m in history if self._seq_of(m) >= anchor]

        if _fits(retained):
            return retained

        # Overflow: advance the anchor in blocks until we are comfortably
        # under budget, so the next few turns do not evict again.
        target = int(budget * config.evict_target_ratio)
        block = max(2, config.evict_block_messages)

        while retained and self._history_tokens(retained) > target:
            # Evict a whole block, then realign so the window still starts on
            # a 'user' turn — a window starting mid-exchange breaks role
            # alternation and renders badly under most chat templates.
            retained = retained[block:]
            while retained and retained[0]["role"] != "user":
                retained = retained[1:]

        if not retained:
            # Budget cannot hold even one exchange. Keep the final user turn
            # rather than returning nothing, and let the caller's reserved
            # response budget absorb the overrun.
            tail = [m for m in history if m["role"] == "user"]
            retained = tail[-1:] if tail else []

        new_anchor = self._seq_of(retained[0]) if retained else anchor
        if new_anchor > anchor:
            await message_cache.set_window_start(conversation_id, new_anchor)

        return retained

    @staticmethod
    def _seq_of(msg: dict) -> int:
        """
        Monotonic ordinal for a message.

        Falls back to 0 when absent so that pre-migration rows, or cache
        entries written before 002_message_seq.sql, degrade to 'include
        everything' rather than raising.
        """
        seq = msg.get("seq")
        return int(seq) if seq is not None else 0

    def _history_tokens(self, msgs: list[dict]) -> int:
        return sum(
            m.get("token_count") or self.count_tokens(m["content"]) for m in msgs
        )

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

    def _format_rag_block(self, chunks: list[dict]) -> str:
        """
        Wrap retrieved chunks for injection immediately before the user turn.

        This block now sits adjacent to the user's question rather than near
        the top of the prompt, so it is delimited explicitly: a trailing
        system message with no closing marker is easy for the model to read
        as part of the conversation instead of as supplied reference material.
        """
        return (
            "Relevant context from documents:\n\n"
            f"{self._format_rag_chunks(chunks)}\n\n"
            "--- end of retrieved context ---\n"
            "Use the retrieved context above when it is relevant to the "
            "user's next message. If it does not contain the answer, say so "
            "rather than guessing."
        )

    async def _get_history(self, conversation_id: str) -> list[dict]:
        """Get conversation history from cache or DB."""
        # Try Redis first
        cached = await message_cache.get_recent(conversation_id)
        if cached is not None:
            return cached

        # Cache miss — load from DB and warm cache.
        # `seq` must survive into the cached entries: it is the anchor the
        # history window is pinned to, and without it a cache hit and a cache
        # miss would select different windows for the same conversation.
        messages = await db.get_messages(conversation_id)
        history = [
            {
                "role": m["role"],
                "content": m["content"],
                "token_count": m.get("token_count"),
                "seq": m.get("seq"),
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
