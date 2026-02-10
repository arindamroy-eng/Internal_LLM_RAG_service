"""Per-user, per-agent vector DB management using Qdrant."""

import logging
from typing import Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings

logger = logging.getLogger(__name__)


class VectorDBManager:
    """
    Manages per-user Qdrant collections for RAG.

    Strategy: one collection per user (user_{user_id}), with
    agent_id filtering within the collection. This gives hard
    isolation between users while allowing agents to share or
    scope documents within a user's namespace.
    """

    def __init__(self):
        self.client: Optional[QdrantClient] = None

    def connect(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )

    def disconnect(self):
        if self.client:
            self.client.close()

    def _collection_name(self, user_id: str) -> str:
        return f"user_{str(user_id).replace('-', '')}"

    def ensure_collection(self, user_id: str) -> str:
        """Create a user's collection if it doesn't exist."""
        name = self._collection_name(user_id)
        collections = [c.name for c in self.client.get_collections().collections]

        if name not in collections:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                ),
            )
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=name,
                field_name="agent_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=name,
                field_name="source_file",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created Qdrant collection: {name}")

        return name

    def upsert_chunks(
        self,
        user_id: str,
        agent_id: str,
        document_id: str,
        chunks: list[dict],
    ):
        """
        Insert document chunks into a user's collection.

        Each chunk dict must have: text, embedding, metadata
        """
        collection = self._collection_name(user_id)
        points = [
            PointStruct(
                id=f"{document_id}_{i}",
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "agent_id": str(agent_id),
                    "document_id": document_id,
                    "source_file": chunk.get("metadata", {}).get("filename", ""),
                    "page": chunk.get("metadata", {}).get("page"),
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        # Batch upsert (Qdrant handles batching internally for large sets)
        BATCH = 100
        for start in range(0, len(points), BATCH):
            self.client.upsert(
                collection_name=collection,
                points=points[start : start + BATCH],
            )

        logger.info(f"Upserted {len(points)} chunks to {collection} for agent {agent_id}")

    def query(
        self,
        user_id: str,
        agent_id: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float = 0.35,
    ) -> list[dict]:
        """Search for relevant chunks within a user+agent scope."""
        collection = self._collection_name(user_id)

        try:
            results = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=str(agent_id)),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )

            return [
                {
                    "text": point.payload["text"],
                    "source": point.payload.get("source_file", ""),
                    "page": point.payload.get("page"),
                    "score": point.score,
                    "document_id": point.payload.get("document_id"),
                }
                for point in results.points
            ]
        except Exception as e:
            logger.warning(f"Qdrant query failed for {collection}: {e}")
            return []

    def delete_agent_documents(self, user_id: str, agent_id: str):
        """Remove all documents for a specific agent."""
        collection = self._collection_name(user_id)
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="agent_id",
                                match=MatchValue(value=str(agent_id)),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted agent {agent_id} documents from {collection}")
        except Exception as e:
            logger.warning(f"Failed to delete agent docs: {e}")

    def delete_document(self, user_id: str, document_id: str):
        """Remove all chunks for a specific document."""
        collection = self._collection_name(user_id)
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to delete document chunks: {e}")

    def get_collection_info(self, user_id: str) -> Optional[dict]:
        """Get stats about a user's collection."""
        collection = self._collection_name(user_id)
        try:
            info = self.client.get_collection(collection)
            return {
                "name": collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception:
            return None


vector_db = VectorDBManager()
