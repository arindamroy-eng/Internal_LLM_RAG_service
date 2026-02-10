"""
Document ingestion pipeline: parse → chunk → embed → store in vector DB.

Supports: PDF, DOCX, TXT, Markdown, HTML, CSV
"""

import hashlib
import logging
from io import BytesIO
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.db.database import db
from app.services.llm_client import llm_client
from app.services.vector_db import vector_db

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )

    async def ingest(
        self,
        user_id: str,
        agent_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> dict:
        """
        Full ingestion pipeline:
        1. Parse file to text
        2. Chunk text
        3. Embed chunks
        4. Store in Qdrant
        5. Track in Postgres
        """
        doc_id = hashlib.sha256(
            f"{user_id}:{agent_id}:{filename}".encode()
        ).hexdigest()[:16]

        collection_name = vector_db.ensure_collection(user_id)

        # Track document in DB
        doc_record = await db.create_document(
            user_id=user_id,
            agent_id=agent_id,
            collection_name=collection_name,
            filename=filename,
        )

        try:
            # 1. Parse
            text = self._extract_text(filename, file_bytes)
            if not text or not text.strip():
                await db.update_document_status(str(doc_record["id"]), "failed")
                return {"doc_id": doc_id, "filename": filename, "chunks": 0, "status": "failed", "error": "No text extracted"}

            # 2. Chunk
            raw_chunks = self.splitter.split_text(text)
            logger.info(f"Split '{filename}' into {len(raw_chunks)} chunks")

            # 3. Embed in batches
            all_chunks = await self._embed_chunks(raw_chunks, filename)

            # 4. Store in vector DB
            vector_db.upsert_chunks(
                user_id=user_id,
                agent_id=agent_id,
                document_id=doc_id,
                chunks=all_chunks,
            )

            # 5. Update tracking
            await db.update_document_status(
                str(doc_record["id"]), "ready", chunk_count=len(all_chunks)
            )

            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(all_chunks),
                "status": "ready",
            }

        except Exception as e:
            logger.error(f"Ingestion failed for '{filename}': {e}")
            await db.update_document_status(str(doc_record["id"]), "failed")
            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": 0,
                "status": "failed",
                "error": str(e),
            }

    def _extract_text(self, filename: str, file_bytes: bytes) -> str:
        """Extract text from various file formats."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext == "pdf":
            return self._extract_pdf(file_bytes)
        elif ext == "docx":
            return self._extract_docx(file_bytes)
        elif ext in ("txt", "md", "csv"):
            return self._extract_text_file(file_bytes)
        elif ext == "html":
            return self._extract_html(file_bytes)
        else:
            # Try as plain text
            return self._extract_text_file(file_bytes)

    def _extract_pdf(self, file_bytes: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    def _extract_docx(self, file_bytes: bytes) -> str:
        from docx import Document

        doc = Document(BytesIO(file_bytes))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_text_file(self, file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            import chardet
            detected = chardet.detect(file_bytes)
            encoding = detected.get("encoding", "utf-8")
            return file_bytes.decode(encoding, errors="replace")

    def _extract_html(self, file_bytes: bytes) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_bytes, "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    async def _embed_chunks(self, raw_chunks: list[str], filename: str) -> list[dict]:
        """Embed text chunks in batches using the embedding model."""
        all_chunks = []
        batch_size = settings.embedding_batch_size

        for i in range(0, len(raw_chunks), batch_size):
            batch_texts = raw_chunks[i : i + batch_size]

            resp = await llm_client.embeddings.create(
                model=settings.embed_model_alias,
                input=batch_texts,
            )

            for j, embedding_data in enumerate(resp.data):
                all_chunks.append(
                    {
                        "text": batch_texts[j],
                        "embedding": embedding_data.embedding,
                        "metadata": {
                            "filename": filename,
                            "chunk_index": i + j,
                        },
                    }
                )

        return all_chunks


ingestion_pipeline = DocumentIngestionPipeline()
