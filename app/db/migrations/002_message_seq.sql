-- ──────────────────────────────────────────────
-- 002: monotonic message sequence
--
-- The context builder needs a stable, monotonic per-message ordinal so the
-- conversation history window can be ANCHORED rather than recomputed each
-- turn. Without an anchor the window slides by one message per turn, which
-- shifts the prompt prefix every turn and drives the vLLM prefix cache hit
-- rate to zero — and prefix-cache hit rate is precisely what the llm-d
-- endpoint picker routes on.
--
-- Why not reuse what already exists:
--   * created_at  — messages are saved in two separate statements per turn
--                   (see _persist_messages in app/routers/chat.py), so two
--                   rows can share a timestamp under load. Not a safe key.
--   * Redis index — MessageCache.append_message LTRIMs to the newest 100,
--                   so positional indices are not stable across turns.
--
-- BIGSERIAL is gapless-enough (gaps on rollback are fine; only monotonicity
-- matters) and backfills existing rows in created_at order.
-- ──────────────────────────────────────────────

ALTER TABLE messages ADD COLUMN IF NOT EXISTS seq BIGSERIAL;

-- Ordering index for the newest-N window query in db.get_messages().
CREATE INDEX IF NOT EXISTS idx_messages_convo_seq
    ON messages(conversation_id, seq ASC);

-- The 001 index ordered by created_at; seq supersedes it for history reads.
DROP INDEX IF EXISTS idx_messages_convo;
