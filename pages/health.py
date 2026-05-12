"""
System health check — for operators and PTs only.

Checks all runtime dependencies and reports their status without
displaying patient data or exposing secret values.

URL: /health  (Streamlit multi-page routing)
Built-in Streamlit liveness: /_stcore/health  (returns HTTP 200 when the
process is alive — useful for uptime monitors and load-balancer probes).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="System Health", layout="centered")

st.title("System Health")
st.caption(f"Checked at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> None:
    col1, col2 = st.columns([3, 7])
    col1.success(label)
    if detail:
        col2.write(detail)


def _warn(label: str, detail: str = "") -> None:
    col1, col2 = st.columns([3, 7])
    col1.warning(label)
    if detail:
        col2.write(detail)


def _fail(label: str, detail: str = "") -> None:
    col1, col2 = st.columns([3, 7])
    col1.error(label)
    if detail:
        col2.write(detail)


# ── Check 1: Database ─────────────────────────────────────────────────────────

st.subheader("1 · Database (SQLite)")
try:
    t0 = time.perf_counter()
    from data.db import DB_PATH, get_db
    with get_db() as db:
        patient_count = db._conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    elapsed = (time.perf_counter() - t0) * 1000
    _ok("Connected", f"`{DB_PATH}` — {patient_count} patient(s) — {elapsed:.1f} ms")
except Exception as exc:
    _fail("Failed", str(exc))

# ── Check 2: ChromaDB index ───────────────────────────────────────────────────

st.subheader("2 · Protocol knowledge base (ChromaDB)")
try:
    t0 = time.perf_counter()
    import chromadb
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection("acl_protocols")
    doc_count = collection.count()
    elapsed = (time.perf_counter() - t0) * 1000

    if doc_count == 0:
        _warn(
            "Empty index",
            f"`{chroma_dir}` — 0 documents. "
            "Run `python -m rag.ingest` before generating plans.",
        )
    else:
        _ok("Indexed", f"`{chroma_dir}` — {doc_count} chunk(s) — {elapsed:.1f} ms")
except Exception as exc:
    _fail("Failed", str(exc))

# ── Check 3: Anthropic API key ────────────────────────────────────────────────

st.subheader("3 · Anthropic API key")
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    _fail("Missing", "Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` or environment.")
elif not api_key.startswith("sk-ant-"):
    _warn("Unexpected format", "Key is set but does not match the expected `sk-ant-` prefix.")
else:
    masked = api_key[:12] + "..." + api_key[-4:]
    _ok("Set", f"`{masked}` ({len(api_key)} chars)")

# ── Check 4: Journal pepper ───────────────────────────────────────────────────

st.subheader("4 · Journal pepper (JOURNAL_PEPPER)")
pepper = os.environ.get("JOURNAL_PEPPER", "")
if not pepper:
    _warn(
        "Not set",
        "Journal entries will use passphrase-only derivation. "
        "Set `JOURNAL_PEPPER` to a 64-hex-char random string for defence in depth. "
        "See `.streamlit/secrets.toml.example` for generation instructions.",
    )
elif len(pepper) < 32:
    _warn("Too short", f"Pepper is {len(pepper)} chars — recommended minimum is 32.")
else:
    _ok("Set", f"{len(pepper)}-char pepper configured.")

# ── Check 5: Environment summary ─────────────────────────────────────────────

st.subheader("5 · Runtime environment")
chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

try:
    from data.db import DB_PATH as _db_path
except Exception:
    _db_path = os.environ.get("DATABASE_URL") or os.environ.get("DB_PATH", "?")

env_rows = {
    "DATABASE_URL / DB_PATH": _db_path,
    "CHROMA_PERSIST_DIR": chroma_dir,
    "ANTHROPIC_API_KEY": "set" if api_key else "not set",
    "JOURNAL_PEPPER": f"set ({len(pepper)} chars)" if pepper else "not set",
}
st.table(env_rows)

# ── Overall banner ────────────────────────────────────────────────────────────

st.divider()

# Collect the verdict by re-evaluating key conditions
db_ok = True
try:
    from data.db import get_db
    with get_db() as db:
        db._conn.execute("SELECT 1")
except Exception:
    db_ok = False

chroma_ok = True
chroma_nonempty = True
try:
    import chromadb as _chromadb
    _col = _chromadb.PersistentClient(
        path=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    ).get_or_create_collection("acl_protocols")
    if _col.count() == 0:
        chroma_nonempty = False
except Exception:
    chroma_ok = False

key_ok = bool(api_key)

if db_ok and chroma_ok and chroma_nonempty and key_ok:
    st.success("All systems operational — app is ready to use.")
elif db_ok and key_ok:
    st.warning(
        "Core services up, but the protocol knowledge base is empty. "
        "Plan generation will fail until `python -m rag.ingest` is run."
    )
else:
    st.error("One or more critical checks failed — see details above.")

st.caption(
    "This page does not display patient data. "
    "Built-in Streamlit liveness probe: `/_stcore/health`"
)
