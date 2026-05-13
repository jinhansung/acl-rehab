"""ChromaDB query wrapper — two public functions used by different callers.

query()               → plain string  (used by state_machine coaching loop)
query_with_metadata() → list of (text, chunk_id, score)  (used by tools.generate_plan)
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

def _resolve_chroma_dir() -> str:
    if os.getenv("CHROMA_PERSIST_DIR"):
        return os.getenv("CHROMA_PERSIST_DIR")
    # On Streamlit Community Cloud the app root is at /mount/src/<repo>
    # Prefer the committed index there; fall back to /tmp for writes
    cloud_committed = "/mount/src/acl-rehab/chroma_db"
    if os.path.exists(cloud_committed):
        return cloud_committed
    if os.path.exists("/mount/src"):
        return "/tmp/chroma_db"
    return "./chroma_db"

CHROMA_DIR = _resolve_chroma_dir()
COLLECTION_NAME = "acl_protocols"
TOP_K = 3


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def query(text: str, protocol: Optional[str] = None, top_k: int = TOP_K) -> str:
    """Return top-k passages concatenated as a plain string (coaching loop)."""
    where = {"protocol_name": protocol} if protocol else None
    try:
        results = _get_collection().query(
            query_texts=[text],
            n_results=top_k,
            where=where,
            include=["documents"],
        )
        docs = results["documents"][0] if results["documents"] else []
        return "\n---\n".join(docs) if docs else "No relevant protocol content found."
    except Exception:
        return "Protocol knowledge base not yet indexed. Run: python -m rag.ingest"


def query_with_metadata(
    text: str,
    protocol: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[tuple[str, str, float]]:
    """
    Return [(document_text, chunk_id, cosine_score), ...] for plan generation.

    chunk_id is the ChromaDB document id — used as rag_source_id in RehabPlan.
    cosine_score is 1 - distance (higher = more similar).
    """
    where = {"protocol_name": protocol} if protocol else None
    try:
        results = _get_collection().query(
            query_texts=[text],
            n_results=top_k,
            where=where,
            include=["documents", "distances", "ids"],
        )
    except Exception:
        return []

    docs = results["documents"][0] if results["documents"] else []
    ids = results["ids"][0] if results["ids"] else []
    distances = results["distances"][0] if results["distances"] else []

    return [
        (doc, chunk_id, round(1.0 - dist, 4))
        for doc, chunk_id, dist in zip(docs, ids, distances)
    ]
