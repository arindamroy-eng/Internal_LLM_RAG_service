"""Agent CRUD endpoints."""

import json
from fastapi import APIRouter, Depends, HTTPException

from app.auth.auth import get_current_user, verify_ownership
from app.db.database import db
from app.models.schemas import AgentCreate, AgentUpdate, AgentResponse
from app.services.vector_db import vector_db

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentResponse)
async def create_agent(body: AgentCreate, user=Depends(get_current_user)):
    """Create a new agent for the authenticated user."""
    data = body.model_dump()
    rag_collection = None

    if body.enable_rag:
        rag_collection = vector_db.ensure_collection(str(user["id"]))

    data["rag_collection"] = rag_collection
    data["tools"] = json.dumps(data.get("tools", []))
    data.pop("enable_rag", None)

    agent = await db.create_agent(str(user["id"]), data)
    agent["id"] = str(agent["id"])
    agent["user_id"] = str(agent["user_id"])
    if isinstance(agent.get("tools"), str):
        agent["tools"] = json.loads(agent["tools"])
    return agent


@router.get("", response_model=list[AgentResponse])
async def list_agents(user=Depends(get_current_user)):
    """List all agents for the authenticated user."""
    agents = await db.list_agents(str(user["id"]))
    for a in agents:
        a["id"] = str(a["id"])
        a["user_id"] = str(a["user_id"])
        if isinstance(a.get("tools"), str):
            a["tools"] = json.loads(a["tools"])
    return agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, user=Depends(get_current_user)):
    """Get a specific agent by ID."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)
    agent["id"] = str(agent["id"])
    agent["user_id"] = str(agent["user_id"])
    if isinstance(agent.get("tools"), str):
        agent["tools"] = json.loads(agent["tools"])
    return agent


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, body: AgentUpdate, user=Depends(get_current_user)):
    """Update an agent's configuration."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    update_data = body.model_dump(exclude_none=True)
    if "tools" in update_data:
        update_data["tools"] = json.dumps(update_data["tools"])

    updated = await db.update_agent(agent_id, update_data)
    updated["id"] = str(updated["id"])
    updated["user_id"] = str(updated["user_id"])
    if isinstance(updated.get("tools"), str):
        updated["tools"] = json.loads(updated["tools"])
    return updated


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str, user=Depends(get_current_user)):
    """Delete an agent and its associated vector data."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    # Clean up vector DB data
    if agent.get("rag_collection"):
        vector_db.delete_agent_documents(str(agent["user_id"]), agent_id)

    await db.delete_agent(agent_id)
    return {"status": "deleted", "agent_id": agent_id}
