"""
Admin — Protocol knowledge base management.
PT-only page: upload PDFs and ingest them into ChromaDB.

The index lives in /tmp/chroma_db on Streamlit Cloud (survives warm restarts
but is rebuilt on cold starts). Upload once per session after a cold start,
or commit a pre-built index to the repo for persistence.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Admin — Knowledge Base", layout="centered")
st.title("Protocol Knowledge Base")

# ── Quick status ──────────────────────────────────────────────────────────────

def _chroma_path() -> str:
    if os.environ.get("CHROMA_PERSIST_DIR"):
        return os.environ["CHROMA_PERSIST_DIR"]
    committed = "/mount/src/acl-rehab/chroma_db"
    if os.path.exists(committed):
        return committed
    if os.path.exists("/mount/src"):
        return "/tmp/chroma_db"
    return "./chroma_db"


@st.cache_resource(show_spinner="Loading embedding model…")
def _get_collection():
    """Load the ChromaDB collection with the embedding model (cached)."""
    import chromadb
    from rag.ingest import NomicEmbedFunction, COLLECTION_NAME
    chroma_dir = _chroma_path()
    client = chromadb.PersistentClient(path=chroma_dir)
    ef = NomicEmbedFunction()
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        # Collection exists with wrong (default) embedding function — recreate it.
        client.delete_collection(COLLECTION_NAME)
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )


def _chunk_and_upsert(pdf_bytes: bytes, filename: str, protocol_name: str) -> int:
    """Chunk a PDF (supplied as bytes) and upsert into the collection."""
    from pypdf import PdfReader
    import re

    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 150

    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks, ids, metas = [], [], []

    stem = Path(filename).stem
    for page_num, page in enumerate(reader.pages, start=1):
        raw = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not raw:
            continue
        start, chunk_idx = 0, 0
        while start < len(raw):
            chunk_text = raw[start : start + CHUNK_SIZE]
            chunk_id = f"{stem}_p{page_num}_c{chunk_idx}"
            chunks.append(chunk_text)
            ids.append(chunk_id)
            metas.append({
                "source_pdf": filename,
                "page_number": page_num,
                "protocol_name": protocol_name,
            })
            chunk_idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    col = _get_collection()
    for i in range(0, len(chunks), 500):
        col.upsert(
            documents=chunks[i : i + 500],
            ids=ids[i : i + 500],
            metadatas=metas[i : i + 500],
        )
    return len(chunks)


# ── Current index status ──────────────────────────────────────────────────────

st.subheader("Current index status")
try:
    import chromadb as _cb
    _client = _cb.PersistentClient(path=_chroma_path())
    _col = _client.get_or_create_collection("acl_protocols")
    doc_count = _col.count()
    if doc_count == 0:
        st.warning(f"Index is empty — upload protocol PDFs below.")
    else:
        st.success(f"{doc_count} chunks indexed at `{_chroma_path()}`")
except Exception as e:
    st.error(f"Could not connect to ChromaDB: {e}")

st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────

st.subheader("Upload protocol PDFs")
st.write(
    "Upload one or more protocol PDFs. Assign each a protocol name so the "
    "RAG retriever can filter by protocol when generating plans."
)

uploaded = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
)

PROTOCOL_OPTIONS = ["MOON", "Delaware-Oslo", "Aspetar", "Other"]

if uploaded:
    st.write(f"**{len(uploaded)} file(s) selected.** Assign protocol names:")
    assignments: dict[str, str] = {}
    for f in uploaded:
        col1, col2 = st.columns([2, 2])
        col1.write(f"`{f.name}`")
        protocol = col2.selectbox(
            "Protocol",
            PROTOCOL_OPTIONS,
            key=f"proto_{f.name}",
            label_visibility="collapsed",
        )
        assignments[f.name] = protocol

    if st.button("Ingest into knowledge base", type="primary"):
        progress = st.progress(0, text="Starting…")
        total_chunks = 0
        errors = []

        for idx, f in enumerate(uploaded):
            protocol_name = assignments[f.name]
            progress.progress(
                idx / len(uploaded),
                text=f"Ingesting `{f.name}` as **{protocol_name}**…",
            )
            try:
                pdf_bytes = f.read()
                n = _chunk_and_upsert(pdf_bytes, f.name, protocol_name)
                total_chunks += n
                st.write(f"✓ `{f.name}` → {n} chunks")
            except Exception as e:
                errors.append((f.name, str(e)))
                st.write(f"✗ `{f.name}` — {e}")

        progress.progress(1.0, text="Done.")

        if errors:
            st.error(f"{len(errors)} file(s) failed — see above.")
        else:
            st.success(
                f"Ingested {total_chunks} chunks from {len(uploaded)} file(s). "
                "Plan generation is now available."
            )
            st.info(
                "**Note:** this index lives in `/tmp/` on Streamlit Cloud and will be "
                "cleared on a cold restart. To make it permanent, run "
                "`python -m rag.ingest` locally, commit `chroma_db/`, and push."
            )
            st.cache_resource.clear()
            st.rerun()

st.divider()

# ── Danger zone ───────────────────────────────────────────────────────────────

with st.expander("Danger zone"):
    st.warning("This clears the entire knowledge base. All chunks will be deleted.")
    if st.button("Clear index", type="secondary"):
        try:
            import chromadb as _cb2
            _client2 = _cb2.PersistentClient(path=_chroma_path())
            _client2.delete_collection("acl_protocols")
            st.cache_resource.clear()
            st.success("Index cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
