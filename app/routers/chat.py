"""Chat endpoint with streaming support and RAG integration."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth.auth import get_current_user, verify_ownership
from app.config import settings
from app.db.database import db
from app.models.schemas import ChatRequest, ChatResponse
from app.services.context_manager import context_manager
from app.services.llm_client import llm_client
from app.services.message_cache import message_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents/{agent_id}/chat", tags=["chat"])


@router.post("")
async def chat(agent_id: str, body: ChatRequest, user=Depends(get_current_user)):
    """
    Send a message to an agent and get a response.

    - Automatically creates a new conversation if conversation_id is not provided
    - Retrieves relevant documents via RAG if the agent has it enabled
    - Assembles conversation history within the token budget
    - Streams the response via Server-Sent Events (SSE)
    """
    # Verify agent ownership
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    # Get or create conversation
    conversation_id = body.conversation_id
    if not conversation_id:
        convo = await db.create_conversation(agent_id, str(user["id"]))
        conversation_id = str(convo["id"])
    else:
        convo = await db.get_conversation(conversation_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
        verify_ownership(str(convo["user_id"]), user)

    # Build context window (system prompt + RAG + history + new message)
    try:
        messages = await context_manager.build_context(
            agent_id=agent_id,
            conversation_id=conversation_id,
            new_user_message=body.message,
        )
    except Exception as e:
        logger.error(f"Context build failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to build context")

    if body.stream:
        return StreamingResponse(
            _stream_response(
                agent=agent,
                messages=messages,
                conversation_id=conversation_id,
                user_message=body.message,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-Id": conversation_id,
            },
        )
    else:
        return await _sync_response(
            agent=agent,
            messages=messages,
            conversation_id=conversation_id,
            user_message=body.message,
        )


async def _stream_response(
    agent: dict,
    messages: list[dict],
    conversation_id: str,
    user_message: str,
):
    """Stream LLM response via SSE."""
    full_response = ""

    # Send conversation_id first
    yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"

    try:
        stream = await llm_client.chat.completions.create(
            model=agent["model"],
            messages=messages,
            temperature=agent.get("temperature", 0.7),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id})}\n\n"

    except Exception as e:
        logger.error(f"LLM streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return

    # Persist messages after streaming completes
    await _persist_messages(conversation_id, user_message, full_response)


async def _sync_response(
    agent: dict,
    messages: list[dict],
    conversation_id: str,
    user_message: str,
) -> ChatResponse:
    """Non-streaming LLM response."""
    try:
        response = await llm_client.chat.completions.create(
            model=agent["model"],
            messages=messages,
            temperature=agent.get("temperature", 0.7),
            stream=False,
        )

        assistant_message = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else None

        # Persist messages
        await _persist_messages(conversation_id, user_message, assistant_message)

        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
            model=agent["model"],
            usage=usage,
        )

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")


async def _persist_messages(
    conversation_id: str, user_message: str, assistant_message: str
):
    """Save messages to DB and update Redis cache."""
    try:
        # Save to Postgres
        await db.save_message(conversation_id, "user", user_message)
        await db.save_message(conversation_id, "assistant", assistant_message)

        # Update Redis cache
        await message_cache.append_message(
            conversation_id, {"role": "user", "content": user_message}
        )
        await message_cache.append_message(
            conversation_id, {"role": "assistant", "content": assistant_message}
        )
    except Exception as e:
        logger.error(f"Failed to persist messages: {e}")
