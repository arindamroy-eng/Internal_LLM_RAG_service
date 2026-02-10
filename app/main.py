"""
Internal LLM RAG Service — Main Application

Multi-tenant LLM platform with per-user RAG, served on 8× MI350X GPUs.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.database import db
from app.services.message_cache import message_cache
from app.services.vector_db import vector_db
from app.auth.auth import get_current_user, require_admin, generate_api_key, hash_api_key
from app.models.schemas import UserCreate, UserResponse, HealthResponse

from app.routers import agents, conversations, documents, chat

# ── Logging ──
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Application Lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    # Startup
    logger.info("Starting Internal LLM RAG Service...")

    logger.info("Connecting to PostgreSQL...")
    await db.connect()

    logger.info("Connecting to Redis...")
    await message_cache.connect()

    logger.info("Connecting to Qdrant...")
    vector_db.connect()

    logger.info("All services connected. Ready to serve.")
    yield

    # Shutdown
    logger.info("Shutting down...")
    await db.disconnect()
    await message_cache.disconnect()
    vector_db.disconnect()
    logger.info("Shutdown complete.")


# ── FastAPI App ──
app = FastAPI(
    title=settings.app_name,
    description="Self-hosted multi-tenant LLM platform with per-user RAG on 8× MI350X GPUs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──
api_prefix = "/api/v1"
app.include_router(agents.router, prefix=api_prefix)
app.include_router(conversations.router, prefix=api_prefix)
app.include_router(documents.router, prefix=api_prefix)
app.include_router(chat.router, prefix=api_prefix)


# ── Admin: User Management ──
@app.post(f"{api_prefix}/admin/users", response_model=UserResponse, tags=["admin"])
async def create_user(body: UserCreate, admin=Depends(require_admin)):
    """Create a new user and generate their API key. (Admin only)"""
    # Check if username exists
    existing = await db.get_user_by_username(body.username)
    if existing:
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Username already exists")

    api_key = generate_api_key()
    api_key_hash = hash_api_key(api_key)

    user = await db.create_user(body.username, api_key_hash)
    return UserResponse(
        id=str(user["id"]),
        username=user["username"],
        api_key=api_key,  # Only returned on creation
        created_at=user["created_at"],
    )


@app.post(f"{api_prefix}/admin/users/{{username}}/rotate-key", tags=["admin"])
async def rotate_api_key(username: str, admin=Depends(require_admin)):
    """Rotate a user's API key. (Admin only)"""
    user = await db.get_user_by_username(username)
    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="User not found")

    new_key = generate_api_key()
    new_hash = hash_api_key(new_key)
    await db.pool.execute(
        "UPDATE users SET api_key_hash = $1 WHERE id = $2::uuid",
        new_hash, str(user["id"])
    )
    return {"username": username, "new_api_key": new_key}


# ── Health Check ──
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Check connectivity to all backing services."""
    status = "healthy"
    checks = {}

    # Postgres
    try:
        await db.pool.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        status = "degraded"

    # Redis
    try:
        await message_cache.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        status = "degraded"

    # Qdrant
    try:
        vector_db.client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        status = "degraded"

    # LiteLLM
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"http://{settings.litellm_host}:{settings.litellm_port}/health",
                timeout=5.0,
            )
            checks["litellm"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        checks["litellm"] = f"error: {e}"
        status = "degraded"

    return HealthResponse(status=status, **checks)


@app.get("/", tags=["system"])
async def root():
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
