"""
System health check — operators and PTs only.
Self-contained: no imports from rag/ so sentence-transformers is never loaded here.
"""
from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="System Health", layout="centered")
st.title("System Health")
st.caption(f"Checked at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")


# ── Shared path helpers (inlined — no rag imports) ────────────────────────────

def _db_path() -> str:
    on_cloud = os.path.exists("/mount/src")
    default = "/tmp/acl_rehab.db" if on_cloud else "./acl_rehab.db"
    raw = os.environ.get("DATABASE_URL") or os.environ.get("DB_PATH", default)
    if raw.startswith("sqlite:///"):
        raw = raw[len("sqlite:///"):]
    if on_cloud and raw != ":memory:" and not os.path.isabs(raw):
        raw = "/tmp/" + os.path.basename(raw)
    return raw


def _chroma_path() -> str:
    if os.environ.get("CHROMA_PERSIST_DIR"):
        return os.environ["CHROMA_PERSIST_DIR"]
    committed = "/mount/src/acl-rehab/chroma_db"
    if os.path.exists(committed):
        return committed
    if os.path.exists("/mount/src"):
        return "/tmp/chroma_db"
    return "./chroma_db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(label: str, detail: str = "") -> None:
    c1, c2 = st.columns([3, 7])
    c1.success(label)
    if detail:
        c2.write(detail)


def _warn(label: str, detail: str = "") -> None:
    c1, c2 = st.columns([3, 7])
    c1.warning(label)
    if detail:
        c2.write(detail)


def _fail(label: str, detail: str = "") -> None:
    c1, c2 = st.columns([3, 7])
    c1.error(label)
    if detail:
        c2.code(detail)


# ── 1. Database ───────────────────────────────────────────────────────────────

st.subheader("1 · Database (SQLite)")
_resolved_db = _db_path()
try:
    import sqlite3
    t0 = time.perf_counter()
    conn = sqlite3.connect(_resolved_db)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            side TEXT NOT NULL,
            graft_type TEXT NOT NULL,
            surgery_date TEXT NOT NULL,
            weight_bearing_status TEXT NOT NULL,
            meniscal_repair TEXT NOT NULL,
            stated_goal_text TEXT NOT NULL,
            protocol TEXT NOT NULL,
            pt_code TEXT,
            created_at TEXT NOT NULL
        );
    """)
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    elapsed = (time.perf_counter() - t0) * 1000
    _ok("Connected", f"`{_resolved_db}` — {count} patient(s) — {elapsed:.1f} ms")
except Exception:
    _fail("Failed", traceback.format_exc())

# ── 2. ChromaDB ───────────────────────────────────────────────────────────────

st.subheader("2 · Protocol knowledge base (ChromaDB)")
_resolved_chroma = _chroma_path()
try:
    import chromadb
    t0 = time.perf_counter()
    client = chromadb.PersistentClient(path=_resolved_chroma)
    col = client.get_or_create_collection("acl_protocols")
    doc_count = col.count()
    elapsed = (time.perf_counter() - t0) * 1000
    if doc_count == 0:
        _warn("Empty index",
              f"`{_resolved_chroma}` — 0 documents. "
              "Add PDFs to `protocols/`, run `python -m rag.ingest`, commit & push.")
    else:
        _ok("Indexed", f"`{_resolved_chroma}` — {doc_count} chunk(s) — {elapsed:.1f} ms")
except Exception:
    _fail("Failed", traceback.format_exc())

# ── 3. Anthropic API key ──────────────────────────────────────────────────────

st.subheader("3 · Anthropic API key")
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    _fail("Missing", "Set ANTHROPIC_API_KEY in Streamlit Cloud secrets.")
elif not api_key.startswith("sk-ant-"):
    _warn("Unexpected format", "Key set but does not start with sk-ant-")
else:
    _ok("Set", f"`{api_key[:12]}...{api_key[-4:]}` ({len(api_key)} chars)")

# ── 4. Journal pepper ─────────────────────────────────────────────────────────

st.subheader("4 · Journal pepper")
pepper = os.environ.get("JOURNAL_PEPPER", "")
if not pepper:
    _warn("Not set", "Passphrase-only encryption active. Fine for development.")
elif len(pepper) < 32:
    _warn("Too short", f"{len(pepper)} chars — recommend 64 hex chars.")
else:
    _ok("Set", f"{len(pepper)}-char pepper configured.")

# ── 5. Environment summary ────────────────────────────────────────────────────

st.subheader("5 · Runtime environment")
st.table({
    "DATABASE_URL resolved": _resolved_db,
    "CHROMA_PERSIST_DIR resolved": _resolved_chroma,
    "On Streamlit Cloud": str(os.path.exists("/mount/src")),
    "ANTHROPIC_API_KEY": "set" if api_key else "not set",
    "JOURNAL_PEPPER": f"{len(pepper)} chars" if pepper else "not set",
})

# ── Overall verdict ───────────────────────────────────────────────────────────

st.divider()
db_live = False
chroma_live = False
try:
    import sqlite3 as _sq
    _sq.connect(_resolved_db).execute("SELECT 1").fetchone()
    db_live = True
except Exception:
    pass

try:
    import chromadb as _cb
    _cb.PersistentClient(path=_resolved_chroma).get_or_create_collection("acl_protocols")
    chroma_live = True
except Exception:
    pass

if db_live and chroma_live and api_key:
    st.success("All systems operational.")
elif db_live and api_key:
    st.warning("Core services up. ChromaDB empty — ingest PDFs to enable plan generation.")
else:
    st.error("One or more critical checks failed — see tracebacks above.")

st.caption("Liveness probe: `/_stcore/health`")
