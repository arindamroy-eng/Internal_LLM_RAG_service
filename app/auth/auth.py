"""Authentication via API keys and JWT tokens."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.config import settings
from app.db.database import db

security = HTTPBearer()


def generate_api_key() -> str:
    """Generate a new API key with 'sk-' prefix."""
    return f"sk-{secrets.token_urlsafe(48)}"


def hash_api_key(api_key: str) -> str:
    """SHA-256 hash of an API key for safe storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_jwt_token(user_id: str, username: str) -> str:
    """Create a JWT token for session-based auth."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiry_hours)
    payload = {
        "sub": user_id,
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """
    Authenticate via Bearer token.
    Supports both API keys (sk-...) and JWT tokens.
    """
    token = credentials.credentials

    # ── Admin key check ──
    if token == settings.admin_api_key:
        return {"id": "admin", "username": "admin", "is_admin": True}

    # ── API key auth (sk-...) ──
    if token.startswith("sk-"):
        key_hash = hash_api_key(token)
        user = await db.get_user_by_api_key_hash(key_hash)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key")
        user["is_admin"] = False
        return user

    # ── JWT auth ──
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user["is_admin"] = False
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Require admin access."""
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def verify_ownership(resource_user_id: str, current_user: dict):
    """Ensure the current user owns the resource."""
    if current_user.get("is_admin"):
        return
    if str(resource_user_id) != str(current_user["id"]):
        raise HTTPException(status_code=403, detail="Access denied")
