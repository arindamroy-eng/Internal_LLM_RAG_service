"""Document upload and management endpoints for RAG."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.auth.auth import get_current_user, verify_ownership
from app.config import settings
from app.db.database import db
from app.models.schemas import DocumentResponse, DocumentIngestResult
from app.services.ingestion import ingestion_pipeline
from app.services.vector_db import vector_db

router = APIRouter(prefix="/agents/{agent_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentIngestResult)
async def upload_document(
    agent_id: str,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """
    Upload a document to an agent's RAG knowledge base.
    Supported formats: PDF, DOCX, TXT, Markdown, HTML, CSV
    """
    # Verify agent ownership
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    if not agent.get("rag_collection"):
        raise HTTPException(
            status_code=400,
            detail="This agent does not have RAG enabled. Recreate with enable_rag=true.",
        )

    # Validate file extension
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}",
        )

    # Validate file size
    file_bytes = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb} MB",
        )

    # Run ingestion pipeline
    result = await ingestion_pipeline.ingest(
        user_id=str(user["id"]),
        agent_id=agent_id,
        filename=file.filename,
        file_bytes=file_bytes,
    )

    if result["status"] == "failed":
        raise HTTPException(status_code=422, detail=result.get("error", "Ingestion failed"))

    return result


@router.get("", response_model=list[DocumentResponse])
async def list_documents(agent_id: str, user=Depends(get_current_user)):
    """List all documents uploaded to this agent."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    docs = await db.list_documents(str(user["id"]), agent_id)
    for d in docs:
        d["id"] = str(d["id"])
        d["user_id"] = str(d["user_id"])
        d["agent_id"] = str(d["agent_id"]) if d.get("agent_id") else None
    return docs


@router.delete("/{document_id}")
async def delete_document(
    agent_id: str, document_id: str, user=Depends(get_current_user)
):
    """Delete a document and its vector embeddings."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    doc = await db.delete_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors
    vector_db.delete_document(str(user["id"]), document_id)

    return {"status": "deleted", "document_id": document_id}


@router.get("/stats")
async def document_stats(agent_id: str, user=Depends(get_current_user)):
    """Get vector DB stats for this agent's collection."""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    verify_ownership(str(agent["user_id"]), user)

    info = vector_db.get_collection_info(str(user["id"]))
    docs = await db.list_documents(str(user["id"]), agent_id)

    return {
        "total_documents": len(docs),
        "total_chunks": sum(d.get("chunk_count", 0) for d in docs),
        "collection": info,
    }
