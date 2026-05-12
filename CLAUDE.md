# ACL Rehab Companion — Claude Code Instructions

## Project overview
Local-first agentic rehab companion. Python + Streamlit frontend,
ChromaDB RAG, Claude API for plan generation only (with explicit consent).

## Architecture rules — never violate
- Journal data (journal.py) NEVER passes through any outbound code path.
  If you touch journal.py, add a comment confirming no server call is made.
- The state machine in agent/state_machine.py is deterministic — no open
  ReAct loops. Each step has explicit allowed_tools and allowed_outputs.
- Cloud model calls (anthropic SDK) are ONLY in agent/tools.py::generate_plan()
  and agent/tools.py::generate_weekly_summary(). Nowhere else.

## Personas — always keep in mind
- Maya: wants hard numbers, honest timelines, no "you're behind"
- Helen: split sessions, "today is hard" button, no streaks
- Sophia: teen, tiered parent view, private journal, may hide volume
When writing any patient-facing copy, run it through the tone rules in
agent/prompts.py::TONE_RULES before finalising.

## Banned words in patient-facing copy
"behind", "should", "!", "you need to", "you must"

## Test commands
- pytest tests/ -v
- python rag/eval.py --run-gold-set    # retrieval quality check
- python tests/persona_traces/run_all.py

## Key files to read before editing
- data/models.py  (Pydantic schemas — don't break them)
- agent/state_machine.py  (step entry conditions)
- agent/red_flags.py  (rules — changes need PT review)

## Running locally
```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY and JOURNAL_KEY
python -m rag.ingest        # index PDFs before first run
streamlit run app.py
```

## Environment variables (see .env.example)
- `ANTHROPIC_API_KEY` — required
- `CHROMA_PERSIST_DIR` — default `./chroma_db`
- `DB_PATH` — default `./acl_rehab.db`
- `JOURNAL_KEY` — Fernet key (generate with `python -m data.journal keygen`)
