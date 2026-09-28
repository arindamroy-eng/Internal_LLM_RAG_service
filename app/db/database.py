"""Async PostgreSQL database layer."""

import asyncpg
from typing import Optional
from app.config import settings


class Database:
    """Async connection pool manager for PostgreSQL."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=10,
            max_size=50,
        )

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    # ──────────────────────────────────
    # Users
    # ──────────────────────────────────

    async def create_user(self, username: str, api_key_hash: str) -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO users (username, api_key_hash)
               VALUES ($1, $2)
               RETURNING id, username, created_at""",
            username, api_key_hash
        )
        return dict(row)

    async def get_user_by_api_key_hash(self, api_key_hash: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT id, username, created_at FROM users WHERE api_key_hash = $1",
            api_key_hash
        )
        return dict(row) if row else None

    async def get_user_by_username(self, username: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT id, username, api_key_hash, created_at FROM users WHERE username = $1",
            username
        )
        return dict(row) if row else None

    async def get_user_by_id(self, user_id: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT id, username, created_at FROM users WHERE id = $1::uuid",
            user_id
        )
        return dict(row) if row else None

    # ──────────────────────────────────
    # Agents
    # ──────────────────────────────────

    async def create_agent(self, user_id: str, data: dict) -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO agents (user_id, name, system_prompt, model, temperature, tools, rag_collection, max_context_tokens)
               VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8)
               RETURNING id, user_id, name, system_prompt, model, temperature, tools, rag_collection, max_context_tokens, created_at""",
            user_id,
            data["name"],
            data["system_prompt"],
            # Default sourced from config rather than hardcoded, so changing
            # CHAT_MODEL_ALIAS does not silently leave this layer behind.
            data.get("model") or settings.chat_model_alias,
            data.get("temperature", 0.7),
            data.get("tools", "[]"),
            data.get("rag_collection"),
            data.get("max_context_tokens", 16384),
        )
        return dict(row)

    async def get_agent(self, agent_id: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT * FROM agents WHERE id = $1::uuid", agent_id
        )
        return dict(row) if row else None

    async def list_agents(self, user_id: str) -> list[dict]:
        rows = await self.pool.fetch(
            "SELECT * FROM agents WHERE user_id = $1::uuid ORDER BY created_at DESC",
            user_id
        )
        return [dict(r) for r in rows]

    async def update_agent(self, agent_id: str, data: dict) -> Optional[dict]:
        sets = []
        values = []
        idx = 1
        for key in ["name", "system_prompt", "model", "temperature", "tools", "rag_collection", "max_context_tokens"]:
            if key in data:
                if key == "tools":
                    sets.append(f"{key} = ${idx}::jsonb")
                else:
                    sets.append(f"{key} = ${idx}")
                values.append(data[key])
                idx += 1
        if not sets:
            return await self.get_agent(agent_id)
        values.append(agent_id)
        query = f"UPDATE agents SET {', '.join(sets)} WHERE id = ${idx}::uuid RETURNING *"
        row = await self.pool.fetchrow(query, *values)
        return dict(row) if row else None

    async def delete_agent(self, agent_id: str) -> bool:
        result = await self.pool.execute(
            "DELETE FROM agents WHERE id = $1::uuid", agent_id
        )
        return result == "DELETE 1"

    # ──────────────────────────────────
    # Conversations
    # ──────────────────────────────────

    async def create_conversation(self, agent_id: str, user_id: str, title: str = None) -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO conversations (agent_id, user_id, title)
               VALUES ($1::uuid, $2::uuid, $3)
               RETURNING id, agent_id, user_id, title, created_at, updated_at""",
            agent_id, user_id, title
        )
        return dict(row)

    async def get_conversation(self, conversation_id: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "SELECT * FROM conversations WHERE id = $1::uuid", conversation_id
        )
        return dict(row) if row else None

    async def list_conversations(self, user_id: str, agent_id: str = None, limit: int = 50) -> list[dict]:
        if agent_id:
            rows = await self.pool.fetch(
                """SELECT * FROM conversations
                   WHERE user_id = $1::uuid AND agent_id = $2::uuid
                   ORDER BY updated_at DESC LIMIT $3""",
                user_id, agent_id, limit
            )
        else:
            rows = await self.pool.fetch(
                """SELECT * FROM conversations
                   WHERE user_id = $1::uuid
                   ORDER BY updated_at DESC LIMIT $2""",
                user_id, limit
            )
        return [dict(r) for r in rows]

    async def delete_conversation(self, conversation_id: str) -> bool:
        result = await self.pool.execute(
            "DELETE FROM conversations WHERE id = $1::uuid", conversation_id
        )
        return result == "DELETE 1"

    # ──────────────────────────────────
    # Messages
    # ──────────────────────────────────

    async def save_message(self, conversation_id: str, role: str, content: str, token_count: int = None, metadata: str = "{}") -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO messages (conversation_id, role, content, token_count, metadata)
               VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
               RETURNING id, conversation_id, role, content, token_count, metadata, created_at, seq""",
            conversation_id, role, content, token_count, metadata
        )
        # Update conversation timestamp
        await self.pool.execute(
            "UPDATE conversations SET updated_at = now() WHERE id = $1::uuid",
            conversation_id
        )
        return dict(row)

    async def get_messages(self, conversation_id: str, limit: int = 100) -> list[dict]:
        """
        Return the NEWEST `limit` messages, in chronological order.

        This previously did `ORDER BY created_at ASC LIMIT $2`, which returns
        the OLDEST messages. For any conversation longer than the limit, a
        Redis cache miss — which happens on TTL expiry or a Redis restart —
        silently rewound the model to the beginning of the conversation.

        The default matches MessageCache.max so the cache-hit and cache-miss
        paths produce the same list; if they diverge, the prompt prefix
        changes depending on cache state and prefix-cache hit rates become
        unreproducible.
        """
        rows = await self.pool.fetch(
            """SELECT id, role, content, token_count, metadata, created_at, seq
               FROM (
                   SELECT id, role, content, token_count, metadata, created_at, seq
                   FROM messages
                   WHERE conversation_id = $1::uuid
                   ORDER BY seq DESC
                   LIMIT $2
               ) recent
               ORDER BY seq ASC""",
            conversation_id, limit
        )
        return [dict(r) for r in rows]

    # ──────────────────────────────────
    # Documents
    # ──────────────────────────────────

    async def create_document(self, user_id: str, agent_id: str, collection_name: str,
                               filename: str, chunk_count: int = 0, status: str = "processing") -> dict:
        row = await self.pool.fetchrow(
            """INSERT INTO documents (user_id, agent_id, collection_name, filename, chunk_count, status)
               VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6)
               RETURNING *""",
            user_id, agent_id, collection_name, filename, chunk_count, status
        )
        return dict(row)

    async def update_document_status(self, doc_id: str, status: str, chunk_count: int = None):
        if chunk_count is not None:
            await self.pool.execute(
                "UPDATE documents SET status = $1, chunk_count = $2 WHERE id = $3::uuid",
                status, chunk_count, doc_id
            )
        else:
            await self.pool.execute(
                "UPDATE documents SET status = $1 WHERE id = $2::uuid",
                status, doc_id
            )

    async def list_documents(self, user_id: str, agent_id: str = None) -> list[dict]:
        if agent_id:
            rows = await self.pool.fetch(
                "SELECT * FROM documents WHERE user_id = $1::uuid AND agent_id = $2::uuid ORDER BY created_at DESC",
                user_id, agent_id
            )
        else:
            rows = await self.pool.fetch(
                "SELECT * FROM documents WHERE user_id = $1::uuid ORDER BY created_at DESC",
                user_id
            )
        return [dict(r) for r in rows]

    async def delete_document(self, doc_id: str) -> Optional[dict]:
        row = await self.pool.fetchrow(
            "DELETE FROM documents WHERE id = $1::uuid RETURNING *", doc_id
        )
        return dict(row) if row else None


db = Database()
