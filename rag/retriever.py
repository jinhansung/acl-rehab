"""ChromaDB query wrapper — two public functions used by different callers.

query()               → plain string  (used by state_machine coaching loop)
query_with_metadata() → list of (text, chunk_id, score)  (used by tools.generate_plan)
"""
from __future__ import annotations
import os
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

def _resolve_chroma_dir() -> str:
    if os.getenv("CHROMA_PERSIST_DIR"):
        return os.getenv("CHROMA_PERSIST_DIR")
    cloud_committed = "/mount/src/acl-rehab/chroma_db"
    if os.path.exists(cloud_committed):
        return cloud_committed
    if os.path.exists("/mount/src"):
        return "/tmp/chroma_db"
    return "./chroma_db"

CHROMA_DIR = _resolve_chroma_dir()
COLLECTION_NAME = "acl_protocols"
TOP_K = 3


def _get_collection():
    """Always returns a fresh collection handle — no caching to avoid stale HNSW index."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def query(text: str, protocol: Optional[str] = None, top_k: int = TOP_K) -> str:
    """Return top-k passages concatenated as a plain string (coaching loop)."""
    where = {"protocol_name": {"$eq": protocol}} if protocol else None
    try:
        col = _get_collection()
        count = col.count()
        if count == 0:
            return "Protocol knowledge base not yet indexed."
        n = min(top_k, count)
        results = col.query(
            query_texts=[text],
            n_results=n,
            where=where,
            include=["documents"],
        )
        docs = results["documents"][0] if results["documents"] else []
        return "\n---\n".join(docs) if docs else "No relevant protocol content found."
    except Exception as exc:
        return f"RAG query error: {exc}"


def query_with_metadata(
    text: str,
    protocol: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[tuple[str, str, float]]:
    """
    Return [(document_text, chunk_id, cosine_score), ...] for plan generation.

    Raises RuntimeError with a descriptive message on failure so callers
    can surface the actual problem rather than silently getting empty results.
    """
    where = {"protocol_name": {"$eq": protocol}} if protocol else None
    col = _get_collection()
    count = col.count()
    if count == 0:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' is empty (path: {CHROMA_DIR}). "
            "Ingest protocol PDFs on the Admin page first."
        )
    n = min(top_k, count)
    results = col.query(
        query_texts=[text],
        n_results=n,
        where=where,
        include=["documents", "distances"],
    )

    docs      = results["documents"][0] if results["documents"] else []
    ids       = results["ids"][0]       if results["ids"]       else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        (doc, chunk_id, round(1.0 - dist, 4))
        for doc, chunk_id, dist in zip(docs, ids, distances)
    ]
