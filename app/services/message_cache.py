"""Redis-backed message cache for fast conversation history retrieval."""

import json
from typing import Optional

import redis.asyncio as redis
from app.config import settings


class MessageCache:
    """
    Caches recent messages per conversation in Redis.
    Avoids hitting Postgres on every LLM call.
    """

    def __init__(self, max_cached_messages: int = 100):
        self.redis: Optional[redis.Redis] = None
        self.max = max_cached_messages

    async def connect(self):
        self.redis = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def disconnect(self):
        if self.redis:
            await self.redis.close()

    def _key(self, conversation_id: str) -> str:
        return f"convo:{conversation_id}:messages"

    def _summary_key(self, conversation_id: str) -> str:
        return f"convo:{conversation_id}:summary"

    async def append_message(self, conversation_id: str, message: dict):
        """Add a message to the conversation cache."""
        key = self._key(conversation_id)
        await self.redis.rpush(key, json.dumps(message))
        await self.redis.ltrim(key, -self.max, -1)
        await self.redis.expire(key, settings.redis_message_ttl)

    async def get_recent(self, conversation_id: str) -> Optional[list[dict]]:
        """Get cached messages. Returns None on cache miss."""
        key = self._key(conversation_id)
        raw = await self.redis.lrange(key, 0, -1)
        if raw:
            return [json.loads(m) for m in raw]
        return None

    async def warm_cache(self, conversation_id: str, messages: list[dict]):
        """Populate cache from DB (on cache miss)."""
        key = self._key(conversation_id)
        pipe = self.redis.pipeline()
        pipe.delete(key)
        for msg in messages[-self.max:]:
            pipe.rpush(key, json.dumps(msg))
        pipe.expire(key, settings.redis_message_ttl)
        await pipe.execute()

    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """Get cached conversation summary."""
        return await self.redis.get(self._summary_key(conversation_id))

    async def set_summary(self, conversation_id: str, summary: str):
        """Cache a conversation summary."""
        await self.redis.setex(
            self._summary_key(conversation_id),
            settings.redis_summary_ttl,
            summary,
        )

    async def invalidate(self, conversation_id: str):
        """Clear cache for a conversation."""
        pipe = self.redis.pipeline()
        pipe.delete(self._key(conversation_id))
        pipe.delete(self._summary_key(conversation_id))
        await pipe.execute()


message_cache = MessageCache()
