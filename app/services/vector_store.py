"""
app/services/vector_store.py
=============================
Manages ChromaDB as the vector database for storing and retrieving
document chunk embeddings.

Storage design:
- Each uploaded document gets its own ChromaDB collection: "doc_{id}".
- This makes per-document operations (delete, search within) fast and clean.
- Cross-document search queries all relevant collections and merges results.

ChromaDB stores:
- The embedding vector (for similarity search)
- The chunk text (returned with search results, no need to look up elsewhere)
- Metadata (page_number, chunk_index — for source citation)

Uses PersistentClient so data survives server restarts.
"""

import logging
from typing import List, Dict, Optional

import chromadb

from app.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# VECTOR STORE SERVICE CLASS
# ─────────────────────────────────────────────

class VectorStoreService:
    """
    ChromaDB-backed vector store for document chunk storage and retrieval.

    Each document is stored in its own collection ("doc_{id}"), allowing
    fast per-document operations while supporting cross-document search.

    Usage:
        store = VectorStoreService()
        store.add_document(doc_id=1, chunks=[...], embeddings=[[...]])
        results = store.search(query_embedding=[...], doc_ids=[1], top_k=5)
    """

    def __init__(self) -> None:
        """
        Initialize the ChromaDB PersistentClient.

        PersistentClient stores data on disk at settings.chroma_dir,
        so embeddings survive server restarts without re-processing.
        """
        settings = get_settings()
        self._chroma_dir = settings.chroma_dir

        # PersistentClient writes to disk automatically.
        # path= sets the directory for the SQLite + parquet files.
        self._client = chromadb.PersistentClient(path=self._chroma_dir)

        logger.info(
            f"VectorStoreService initialized with ChromaDB at '{self._chroma_dir}'"
        )

    def _get_collection_name(self, doc_id: int) -> str:
        """
        Generate a consistent collection name for a document ID.

        Args:
            doc_id: The document's database ID.

        Returns:
            Collection name string like "doc_42".
        """
        return f"doc_{doc_id}"

    def add_document(
        self,
        doc_id: int,
        chunks: List[Dict],
        embeddings: List[List[float]],
        document_name: str = "",
    ) -> None:
        """
        Store a document's chunks and their embeddings in ChromaDB.

        Creates (or gets) a collection named "doc_{doc_id}" and upserts
        all chunks with their embeddings and metadata.

        Args:
            doc_id:     The document's database ID (used as collection name).
            chunks:     List of chunk dicts from chunking_service. Each must have:
                        - text (str)
                        - page_number (int)
                        - chunk_index (int)
                        - metadata (dict)
            embeddings: List of embedding vectors, one per chunk.
                        Must be same length as chunks.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
            RuntimeError: If ChromaDB operation fails.
        """
        # ── Validation ────────────────────────
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings. "
                f"These must be the same length."
            )

        if not chunks:
            logger.warning(f"add_document() called with empty chunks for doc_id={doc_id}")
            return

        collection_name = self._get_collection_name(doc_id)
        logger.info(
            f"Storing {len(chunks)} chunks in collection '{collection_name}'"
        )

        try:
            # get_or_create_collection: creates if new, returns existing if already present.
            # This makes re-processing a document safe (idempotent via upsert below).
            collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"document_id": doc_id},
            )

            # ── Prepare data for ChromaDB ─────
            # ChromaDB expects parallel lists of ids, embeddings, documents, and metadatas.
            ids = [
                f"doc_{doc_id}_chunk_{chunk['chunk_index']}"
                for chunk in chunks
            ]

            documents = [chunk["text"] for chunk in chunks]

            # ChromaDB metadata must be flat (no nested dicts) and values
            # must be str, int, float, or bool.
            metadatas = [
                {
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "char_count": chunk["metadata"].get("char_count", len(chunk["text"])),
                    "document_id": doc_id,
                    "document_name": document_name,
                }
                for chunk in chunks
            ]

            # upsert = insert or update. Safe to call multiple times
            # with the same IDs (e.g. when re-processing a document).
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                f"Successfully stored {len(chunks)} chunks in '{collection_name}' "
                f"(collection now has {collection.count()} items)"
            )

        except Exception as e:
            logger.error(
                f"Failed to store chunks in ChromaDB for doc_id={doc_id}: {e}"
            )
            raise RuntimeError(f"Vector store operation failed: {e}") from e

    def search(
        self,
        query_embedding: List[float],
        doc_ids: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for the most similar chunks to a query embedding.

        If doc_ids is provided, only searches those documents' collections.
        Otherwise, searches ALL document collections and merges results.

        Args:
            query_embedding: The embedding vector of the user's question.
            doc_ids:         Optional list of document IDs to search within.
                             If None, searches all documents.
            top_k:           Number of top results to return.

        Returns:
            List of result dicts, sorted by relevance (best first):
                - document_id (int):  Source document ID
                - text (str):         The chunk text
                - page_number (int):  Source page number
                - score (float):      Similarity score (lower distance = better match)

        Raises:
            RuntimeError: If ChromaDB query fails.
        """
        logger.info(
            f"Searching vector store (doc_ids={doc_ids}, top_k={top_k})"
        )

        # ── Determine which collections to search ──
        if doc_ids:
            collection_names = [self._get_collection_name(did) for did in doc_ids]
        else:
            # Search all collections — list them from ChromaDB
            collection_names = [
                col.name for col in self._client.list_collections()
            ]

        if not collection_names:
            logger.warning("No collections found to search")
            return []

        # ── Query each collection and merge results ──
        all_results: List[Dict] = []

        for col_name in collection_names:
            try:
                collection = self._client.get_collection(name=col_name)
            except Exception:
                # Collection might not exist (e.g. doc was deleted).
                logger.debug(f"Collection '{col_name}' not found, skipping")
                continue

            # Skip empty collections
            if collection.count() == 0:
                logger.debug(f"Collection '{col_name}' is empty, skipping")
                continue

            try:
                # Adjust top_k if collection has fewer items
                effective_k = min(top_k, collection.count())

                # ChromaDB query returns distances (lower = more similar for L2/cosine).
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=effective_k,
                    include=["documents", "metadatas", "distances"],
                )

                # ── Parse ChromaDB response ───
                # results is a dict with lists of lists:
                #   results["documents"] = [["text1", "text2", ...]]
                #   results["metadatas"] = [[{...}, {...}, ...]]
                #   results["distances"] = [[0.5, 0.7, ...]]
                if results and results.get("documents") and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        distance = results["distances"][0][i] if results["distances"] else 1.0

                        all_results.append({
                            "document_id": metadata.get("document_id", 0),
                            "document_name": metadata.get("document_name", ""),
                            "text": results["documents"][0][i],
                            "page_number": metadata.get("page_number", 0),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "score": distance,  # Lower distance = better match
                        })

            except Exception as e:
                logger.error(f"Failed to query collection '{col_name}': {e}")
                # Continue searching other collections rather than failing entirely
                continue

        # ── Sort by score (ascending — lower distance = better match) ──
        all_results.sort(key=lambda x: x["score"])

        # ── Return top-K across all collections ──
        final_results = all_results[:top_k]

        logger.info(
            f"Search returned {len(final_results)} results from "
            f"{len(collection_names)} collections"
        )
        return final_results

    def delete_document(self, doc_id: int) -> None:
        """
        Delete a document's entire collection from ChromaDB.

        This removes all chunks and embeddings for the document.
        Safe to call even if the collection doesn't exist.

        Args:
            doc_id: The document's database ID.
        """
        collection_name = self._get_collection_name(doc_id)
        logger.info(f"Deleting collection '{collection_name}'")

        try:
            self._client.delete_collection(name=collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'")
        except ValueError:
            # ChromaDB raises ValueError if collection doesn't exist
            logger.warning(
                f"Collection '{collection_name}' does not exist, nothing to delete"
            )
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise RuntimeError(f"Failed to delete document vectors: {e}") from e

    def get_collection_count(self) -> int:
        """
        Count the total number of document collections in ChromaDB.

        This tells you how many documents have been processed and stored
        (each document gets one collection).

        Returns:
            Number of collections (= number of stored documents).
        """
        try:
            collections = self._client.list_collections()
            count = len(collections)
            logger.debug(f"ChromaDB has {count} collections")
            return count
        except Exception as e:
            logger.error(f"Failed to count ChromaDB collections: {e}")
            raise RuntimeError(f"Failed to count collections: {e}") from e

    def get_document_chunk_count(self, doc_id: int) -> int:
        """
        Count the number of chunks stored for a specific document.

        Args:
            doc_id: The document's database ID.

        Returns:
            Number of chunks in the document's collection.
            Returns 0 if the collection doesn't exist.
        """
        collection_name = self._get_collection_name(doc_id)
        try:
            collection = self._client.get_collection(name=collection_name)
            return collection.count()
        except Exception:
            return 0


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────
# Created once and reused. Import this in other modules:
#     from app.services.vector_store import get_vector_store

_vector_store: VectorStoreService | None = None


def get_vector_store() -> VectorStoreService:
    """
    Get or create the singleton VectorStoreService instance.

    Using a function (instead of a module-level instance) avoids
    ChromaDB initialization at import time — the service is only
    created when first needed.

    Returns:
        The shared VectorStoreService instance.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
