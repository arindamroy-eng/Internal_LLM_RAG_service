"""Conversation management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.auth.auth import get_current_user, verify_ownership
from app.db.database import db
from app.models.schemas import ConversationCreate, ConversationResponse, MessageResponse
from app.services.message_cache import message_cache

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/{agent_id}", response_model=ConversationResponse)
async def create_conversation(
    agent_id: str, body: ConversationCreate = None, user=Depends(get_current_user)
):
    """Start a new conversation with an agent."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    title = body.title if body else None
    convo = await db.create_conversation(agent_id, str(user["id"]), title)
    convo["id"] = str(convo["id"])
    convo["agent_id"] = str(convo["agent_id"])
    convo["user_id"] = str(convo["user_id"])
    return convo


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    agent_id: str = None, limit: int = 50, user=Depends(get_current_user)
):
    """List conversations, optionally filtered by agent."""
    convos = await db.list_conversations(str(user["id"]), agent_id, limit)
    for c in convos:
        c["id"] = str(c["id"])
        c["agent_id"] = str(c["agent_id"])
        c["user_id"] = str(c["user_id"])
    return convos


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, user=Depends(get_current_user)):
    """Get conversation details."""
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(str(convo["user_id"]), user)
    convo["id"] = str(convo["id"])
    convo["agent_id"] = str(convo["agent_id"])
    convo["user_id"] = str(convo["user_id"])
    return convo


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: str, limit: int = 200, user=Depends(get_current_user)
):
    """Get all messages in a conversation."""
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(str(convo["user_id"]), user)

    messages = await db.get_messages(conversation_id, limit)
    for m in messages:
        m["id"] = str(m["id"])
    return messages


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str, user=Depends(get_current_user)):
    """Delete a conversation and all its messages."""
    convo = await db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    verify_ownership(str(convo["user_id"]), user)

    await message_cache.invalidate(conversation_id)
    await db.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}
