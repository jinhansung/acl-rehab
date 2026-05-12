# Deploying ACL Rehab Companion to Streamlit Community Cloud

## Prerequisites

- GitHub account with the repository pushed to a public or private repo
- Streamlit Community Cloud account — sign up at <https://share.streamlit.io>
- Anthropic API key — obtain at <https://console.anthropic.com/settings/keys>

---

## Step 1 — Prepare the repository

### 1a. Verify `.gitignore`

Ensure these entries exist so secrets and local data are never committed:

```
.streamlit/secrets.toml
.env
*.db
chroma_db/
__pycache__/
*.pyc
```

### 1b. Confirm `requirements.txt` is up to date

```bash
pip freeze > requirements.txt   # or manually keep it pinned
```

The app requires (minimum versions):

```
streamlit>=1.35.0
anthropic>=0.28.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
cryptography>=42.0.0
pydantic>=2.7.0
pypdf>=4.0.0
```

### 1c. Set the main file

The entry point must be `app.py` at the repository root. Confirm it exists
and calls `st.set_page_config` (or delegates to a page that does).

---

## Step 2 — Handle the protocol knowledge base

The ChromaDB index (built by `python -m rag.ingest`) is **not** stored in
the repository. Streamlit Community Cloud has no persistent disk, so the
index is lost on every cold start.

**Option A — Commit a pre-built index (recommended for small corpora)**

```bash
python -m rag.ingest        # index your PDFs locally
git add chroma_db/
git commit -m "Add protocol knowledge base index"
```

Remove `chroma_db/` from `.gitignore` for this workflow, or use a
separate branch that includes the index.

**Option B — Rebuild on startup**

Add an `app.py` top-level block that runs ingest if the collection is empty:

```python
import chromadb, os

_CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
_col = chromadb.PersistentClient(path=_CHROMA_DIR).get_or_create_collection("acl_protocols")
if _col.count() == 0:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "rag.ingest"], check=True)
```

This adds ~2–3 minutes to the cold-start time.

**Option C — External vector store (production)**

Replace the ChromaDB backend in `rag/retriever.py` and `rag/ingest.py` with
a managed service (Pinecone, Weaviate Cloud, Qdrant Cloud). The retriever
interface (`query_with_metadata`) remains unchanged.

---

## Step 3 — Persistent storage

SQLite on Streamlit Community Cloud is **ephemeral** — the database file is
lost when the container restarts (roughly once per day or on each deploy).

For a production deployment where patient data must persist:

| Option | Notes |
|---|---|
| **Supabase** (free tier) | Managed Postgres; add `psycopg2-binary` and rewrite `data/db.py` to use `psycopg2` instead of `sqlite3` |
| **Railway** | Managed Postgres with one-command provision |
| **Neon** | Serverless Postgres; minimal cold-start overhead |
| **Turso** (libSQL) | SQLite-compatible API; minimal code change |

Until `data/db.py` is migrated, the app will work on Community Cloud but
**all patient data, plans, and sessions are lost on each restart**.

---

## Step 4 — Configure secrets in Streamlit dashboard

1. Go to <https://share.streamlit.io> → your app → **Settings** → **Secrets**
2. Paste the contents of `.streamlit/secrets.toml.example` and fill in values:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
DATABASE_URL      = "acl_rehab.db"           # or a Postgres URL
JOURNAL_PEPPER    = "<64-hex-char random string>"
```

Generate `JOURNAL_PEPPER`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Set `JOURNAL_PEPPER` once, before any patient uses the app. Never change it.**

Streamlit exports all secrets as environment variables, so `os.environ.get()`
in `data/journal.py` and `data/db.py` will pick them up automatically.

---

## Step 5 — Deploy

1. Push all changes to GitHub (main branch, or a branch of your choice).
2. Log into <https://share.streamlit.io> and click **New app**.
3. Fill in:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy**. Initial build takes 3–8 minutes (model downloads for
   `nomic-embed-text-v1`).
5. Open `/health` in your deployed app to verify all components are green.

---

## Step 6 — Verify after deployment

Navigate to `<your-app-url>/health` and confirm:

| Check | Expected |
|---|---|
| Database | Connected — 0 patients |
| ChromaDB | ≥ 1 chunk (or "Empty index" if ingest is deferred) |
| Anthropic API key | Set — `sk-ant-...` |
| Journal pepper | Set — 64-char pepper |

The built-in Streamlit liveness endpoint is always available at:
```
https://<your-app>.streamlit.app/_stcore/health
```
It returns HTTP 200 with `{"status":"ok"}` whenever the process is running,
suitable for uptime monitors (UptimeRobot, Betterstack, etc.).

---

## Important caveats

### Clinical and regulatory

This software is a **PT-assist tool**, not a certified medical device.
Before deploying to real patients:

- Confirm compliance with applicable regulations (HIPAA in the US;
  GDPR / MDR in the EU; MDSAP where required).
- Conduct a clinical risk assessment (ISO 14971).
- Ensure the PT review gate (`PlanReviewStatus.PENDING` → `APPROVED`)
  is enforced in your workflow.

### Privacy

- Journal ciphertext is stored in the database. With `JOURNAL_PEPPER` set,
  entries are protected by both passphrase and server secret.
- The `data/journal.py` module never makes a network call.
- Patient names are stored in SQLite only; they are stripped from all
  Anthropic API payloads by `agent/tools.py::_anonymise()`.

### API costs

Plan generation and weekly summaries use `claude-sonnet-4-20250514`.
Estimate: ~$0.03–0.10 per plan generation, ~$0.01 per weekly summary
(varies with RAG context length). Monitor usage at
<https://console.anthropic.com/usage>.
