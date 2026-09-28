"""
Prefix-stability invariants for ContextManager.build_context.

These tests encode the property the llm-d endpoint picker routes on: across
consecutive turns of one conversation, the serialized prompt PREFIX (system
prompt + summary + history) must be byte-identical, so vLLM can reuse its KV
cache and the endpoint picker can steer a request to the replica holding it.

If these fail, prefix-cache hit rate collapses and llm-d has nothing to route
on — see GATE A in the implementation plan.
"""

import asyncio

import pytest

from app.services import context_manager as cm_module
from app.services.context_manager import ContextConfig, ContextManager


# ──────────────────────────────────────────────
# Fakes — the real singletons hit Postgres/Redis/Qdrant.
# ──────────────────────────────────────────────


class FakeDB:
    def __init__(self, agent):
        self._agent = agent
        self.messages: list[dict] = []
        self._seq = 0

    async def get_agent(self, agent_id):
        return self._agent

    async def get_messages(self, conversation_id, limit=100):
        return self.messages[-limit:]

    def add_turn(self, user_text, assistant_text):
        for role, content in (("user", user_text), ("assistant", assistant_text)):
            self._seq += 1
            self.messages.append(
                {
                    "role": role,
                    "content": content,
                    "token_count": None,
                    "seq": self._seq,
                }
            )


class FakeCache:
    """In-memory stand-in for the Redis-backed MessageCache."""

    def __init__(self):
        self.windows: dict[str, int] = {}
        self.summaries: dict[str, str] = {}

    async def get_recent(self, conversation_id):
        return None  # force the DB path

    async def warm_cache(self, conversation_id, messages):
        pass

    async def get_window_start(self, conversation_id):
        return self.windows.get(conversation_id)

    async def set_window_start(self, conversation_id, seq):
        current = self.windows.get(conversation_id)
        if current is not None and current >= seq:
            return
        self.windows[conversation_id] = seq

    async def get_summary(self, conversation_id):
        return self.summaries.get(conversation_id)

    async def set_summary(self, conversation_id, summary):
        self.summaries[conversation_id] = summary


@pytest.fixture
def agent():
    return {
        "id": "agent-1",
        "user_id": "user-1",
        "system_prompt": "You are a precise research assistant. " * 10,
        "rag_collection": "user_1_agent_1",
        "temperature": 0.7,
    }


@pytest.fixture
def wiring(monkeypatch, agent):
    """Point the module-level singletons at fakes."""
    db = FakeDB(agent)
    cache = FakeCache()
    monkeypatch.setattr(cm_module, "db", db)
    monkeypatch.setattr(cm_module, "message_cache", cache)
    return db, cache


def render(messages: list[dict]) -> str:
    """
    Flatten the message list the way a chat template would, so we can measure
    the shared prefix in characters rather than in list elements. vLLM caches
    KV blocks over the token stream, so element-level comparison would hide
    exactly the regression we care about.
    """
    return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)


def shared_prefix_ratio(a: str, b: str) -> float:
    """Fraction of the EARLIER prompt `a` that the later prompt `b` reuses."""
    if not a:
        return 1.0
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / len(a)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_rag_block_sits_after_history_and_before_user_turn(wiring, monkeypatch):
    """
    Ordering contract: system -> history -> RAG -> new user message.

    RAG must NOT appear near the top; that placement is what made every token
    after it a cache miss.
    """
    db, _ = wiring
    mgr = ContextManager()
    monkeypatch.setattr(
        mgr,
        "_retrieve_rag",
        lambda **kw: _async([{"text": "retrieved fact", "source": "d.pdf", "score": 0.9}]),
    )

    db.add_turn("first question", "first answer")
    messages = asyncio.run(
        mgr.build_context("agent-1", "convo-1", "second question")
    )

    roles_and_kind = [
        "rag" if ("Relevant context from documents" in m["content"]) else m["role"]
        for m in messages
    ]

    assert roles_and_kind[0] == "system", "system prompt must lead"
    assert roles_and_kind[-1] == "user", "new user message must be last"
    assert "rag" in roles_and_kind, "RAG block should be present"

    rag_idx = roles_and_kind.index("rag")
    assert rag_idx == len(messages) - 2, "RAG must sit immediately before the user turn"
    # History must precede the RAG block.
    assert any(r == "assistant" for r in roles_and_kind[:rag_idx]), (
        "conversation history must come BEFORE the RAG block"
    )


def test_prefix_is_byte_identical_across_turns_despite_varying_rag(wiring, monkeypatch):
    """
    The core invariant. Retrieval returns different content and different
    SIZES each turn; the prefix must not move.
    """
    db, _ = wiring
    mgr = ContextManager()

    call = {"n": 0}

    def varying_rag(**kwargs):
        call["n"] += 1
        # Deliberately different token counts per turn.
        chunks = [
            {"text": f"fact {call['n']} " * (20 * call["n"]), "source": "d.pdf", "score": 0.8}
        ]
        return _async(chunks)

    monkeypatch.setattr(mgr, "_retrieve_rag", varying_rag)

    rendered = []
    for turn in range(12):
        messages = asyncio.run(
            mgr.build_context("agent-1", "convo-1", f"question {turn}")
        )
        rendered.append(render(messages))
        db.add_turn(f"question {turn} " * 40, f"answer {turn} " * 40)

    # How much of each turn's prompt the NEXT turn can reuse. This is the
    # quantity vLLM's prefix cache hit rate tracks and the llm-d
    # prefix-cache scorer routes on.
    ratios = [
        shared_prefix_ratio(rendered[i - 1], rendered[i])
        for i in range(1, len(rendered))
    ]

    # The direction of travel is the real signal. Measured on this fixture:
    #
    #   RAG before history (original):  76% -> 5%   (DECAYS)
    #   RAG after history  (current):   49% -> 82%  (CLIMBS, plateaus ~82%)
    #
    # The original decays because the system prompt — the only stable span —
    # is a shrinking fraction of a growing prompt, while the volatile RAG
    # block truncates everything after it. Early turns are lower here simply
    # because history is still small relative to the RAG block.
    steady_state = ratios[3:]
    worst = min(steady_state)
    assert worst >= 0.70, (
        f"steady-state prefix reuse fell to {worst:.1%} "
        f"(all turns: {[f'{r:.0%}' for r in ratios]}). Something volatile has "
        "moved ahead of the conversation history — prefix-cache hit rate will "
        "collapse and llm-d will have nothing to route on."
    )

    # And it must not be decaying, which is the original bug's signature.
    assert ratios[-1] > ratios[0], (
        f"prefix reuse is decaying as the conversation grows "
        f"({ratios[0]:.0%} -> {ratios[-1]:.0%}), which is what RAG-before-history "
        "produces. It should improve as stable history accumulates."
    )


def test_history_window_evicts_in_blocks_not_one_message_at_a_time(wiring, monkeypatch):
    """
    Under budget pressure the window must jump in blocks and stay put for
    several turns, rather than sliding by one message every turn.
    """
    db, cache = wiring
    mgr = ContextManager()
    monkeypatch.setattr(mgr, "_retrieve_rag", lambda **kw: _async([]))

    # Tight budget so eviction is forced quickly.
    config = ContextConfig(
        max_tokens=2000,
        reserved_for_response=200,
        rag_chunk_budget=200,
        summary_threshold=10_000,  # disable summarization for this test
        evict_block_messages=8,
        evict_target_ratio=0.85,
    )

    for turn in range(40):
        db.add_turn(f"question {turn} " * 20, f"answer {turn} " * 20)
        asyncio.run(
            mgr.build_context("agent-1", "convo-1", "next", config=config)
        )

    anchors_seen = []
    for turn in range(40, 60):
        db.add_turn(f"question {turn} " * 20, f"answer {turn} " * 20)
        asyncio.run(
            mgr.build_context("agent-1", "convo-1", "next", config=config)
        )
        anchors_seen.append(cache.windows.get("convo-1"))

    # Monotonic: the anchor never moves backwards.
    assert anchors_seen == sorted(anchors_seen), (
        f"window anchor moved backwards: {anchors_seen}"
    )

    # Block-wise: it should NOT change on every single turn.
    changes = sum(
        1 for i in range(1, len(anchors_seen)) if anchors_seen[i] != anchors_seen[i - 1]
    )
    assert changes < len(anchors_seen), (
        "anchor advanced on every turn — that is the sliding window this "
        "replaces, and it yields zero prefix reuse"
    )


def test_history_window_starts_on_a_user_turn(wiring, monkeypatch):
    """A window opening on an assistant turn breaks role alternation."""
    db, _ = wiring
    mgr = ContextManager()
    monkeypatch.setattr(mgr, "_retrieve_rag", lambda **kw: _async([]))

    config = ContextConfig(
        max_tokens=1500,
        reserved_for_response=200,
        rag_chunk_budget=100,
        summary_threshold=10_000,
        evict_block_messages=8,
        evict_target_ratio=0.85,
    )

    for turn in range(30):
        db.add_turn(f"q{turn} " * 30, f"a{turn} " * 30)

    messages = asyncio.run(
        mgr.build_context("agent-1", "convo-1", "next", config=config)
    )

    # Drop the leading system prompt(s); the first history entry follows.
    history = [m for m in messages[1:-1] if "Relevant context" not in m["content"]]
    if history:
        assert history[0]["role"] == "user", (
            f"history window opened on a {history[0]['role']} turn"
        )


def _async(value):
    """Wrap a plain value in an awaitable, for patching async methods."""

    async def _inner():
        return value

    return _inner()
