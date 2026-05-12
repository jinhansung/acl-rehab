"""
PDF chunker + embedder for ACL protocol PDFs.

Chunks at 900 chars / 150-char overlap, page-aware.
Stores in ChromaDB with metadata: source_pdf, page_number, protocol_name.
Embeddings via nomic-embed-text (nomic-ai/nomic-embed-text-v1).
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PROTOCOLS_DIR = Path(__file__).parent.parent / "protocols"
# Streamlit Community Cloud: /mount/src is read-only; write ChromaDB to /tmp
_on_cloud = os.path.exists("/mount/src")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db" if _on_cloud else "./chroma_db")
COLLECTION_NAME = "acl_protocols"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Map PDF stem → canonical protocol name stored in metadata
PROTOCOL_NAMES: dict[str, str] = {
    "MOON": "MOON",
    "Delaware-Oslo": "Delaware-Oslo",
    "Aspetar": "Aspetar",
    "DVT": "DVT",
}


class NomicEmbedFunction(EmbeddingFunction):
    """ChromaDB-compatible wrapper around nomic-embed-text-v1."""

    def __init__(self) -> None:
        # trust_remote_code required for nomic-embed-text
        self._model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True,
        )

    def __call__(self, input: Documents) -> Embeddings:
        # nomic-embed-text expects a task prefix for retrieval documents
        prefixed = [f"search_document: {doc}" for doc in input]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_page(text: str, page_num: int, source_pdf: str, protocol_name: str) -> Iterator[dict]:
    """Yield chunk dicts from a single page with 900/150 sliding window."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return

    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        yield {
            "text": chunk_text,
            "source_pdf": source_pdf,
            "page_number": page_num,
            "protocol_name": protocol_name,
            "chunk_index": chunk_index,
        }
        chunk_index += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP


def _chunk_pdf(path: Path) -> list[dict]:
    stem = path.stem
    protocol_name = PROTOCOL_NAMES.get(stem, stem)
    source_pdf = path.name

    reader = PdfReader(str(path))
    chunks: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        chunks.extend(_chunk_page(raw, page_num, source_pdf, protocol_name))
    return chunks


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_all(protocols_dir: Path = PROTOCOLS_DIR, verbose: bool = True) -> int:
    """Index all PDFs in protocols_dir. Returns total chunk count."""
    pdfs = sorted(protocols_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {protocols_dir}. Add protocol PDFs and re-run.")
        return 0

    ef = NomicEmbedFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for pdf_path in pdfs:
        if verbose:
            print(f"Ingesting {pdf_path.name} …")
        chunks = _chunk_pdf(pdf_path)

        ids = [
            f"{pdf_path.stem}_p{c['page_number']}_c{c['chunk_index']}"
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source_pdf": c["source_pdf"],
                "page_number": c["page_number"],
                "protocol_name": c["protocol_name"],
            }
            for c in chunks
        ]

        # Upsert in batches of 500 to avoid memory spikes
        batch = 500
        for i in range(0, len(chunks), batch):
            collection.upsert(
                documents=documents[i : i + batch],
                ids=ids[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )

        if verbose:
            print(f"  → {len(chunks)} chunks indexed")
        total += len(chunks)

    if verbose:
        print(f"\nIngest complete. {total} total chunks in collection '{COLLECTION_NAME}'.")
    return total


if __name__ == "__main__":
    ingest_all()
