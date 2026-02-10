"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ──────────────────────────────────
# Auth
# ──────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=255)


class UserResponse(BaseModel):
    id: str
    username: str
    api_key: Optional[str] = None  # Only returned on creation
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ──────────────────────────────────
# Agents
# ──────────────────────────────────

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    system_prompt: str = Field(..., min_length=1)
    model: str = Field(default="gpt-4", description="Model alias: gpt-4, gpt-3.5-turbo")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    tools: list = Field(default_factory=list, description="Tool/function definitions")
    enable_rag: bool = Field(default=False, description="Enable document upload and RAG retrieval")
    max_context_tokens: int = Field(default=16384, ge=1024, le=131072)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    tools: Optional[list] = None
    max_context_tokens: Optional[int] = Field(default=None, ge=1024, le=131072)


class AgentResponse(BaseModel):
    id: str
    user_id: str
    name: str
    system_prompt: str
    model: str
    temperature: float
    tools: list
    rag_collection: Optional[str]
    max_context_tokens: int
    created_at: datetime


# ──────────────────────────────────
# Conversations
# ──────────────────────────────────

class ConversationCreate(BaseModel):
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    agent_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


# ──────────────────────────────────
# Messages / Chat
# ──────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    stream: bool = Field(default=True)


class ChatResponse(BaseModel):
    conversation_id: str
    message: str
    model: str
    usage: Optional[dict] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    token_count: Optional[int]
    created_at: datetime


# ──────────────────────────────────
# Documents
# ──────────────────────────────────

class DocumentResponse(BaseModel):
    id: str
    user_id: str
    agent_id: Optional[str]
    collection_name: str
    filename: str
    chunk_count: int
    status: str
    created_at: datetime


class DocumentIngestResult(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    status: str


# ──────────────────────────────────
# Health
# ──────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    postgres: str
    redis: str
    qdrant: str
    litellm: str
