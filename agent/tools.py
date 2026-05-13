"""
Cloud model calls are restricted to two functions in this module:
  - generate_plan()          (claude-sonnet-4-20250514)
  - generate_weekly_summary()  (claude-sonnet-4-20250514)

No other file in this project may import or instantiate anthropic.Anthropic.
See CLAUDE.md — architecture rule.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import anthropic

from agent.prompts import (
    BANNED_WORDS,
    PLAN_GENERATION_SYSTEM,
    PLAN_GENERATION_USER,
    WEEKLY_SUMMARY_SYSTEM,
    WEEKLY_SUMMARY_USER,
    check_tone,
)
from data.db import get_db
from data.models import (
    ConsentRecord,
    ConsentType,
    PatientProfile,
    PlanReviewStatus,
    RehabPlan,
    RedFlagEvent,
    SwellingLevel,
)
from rag.retriever import query as rag_query

MODEL = "claude-sonnet-4-6"

# ── Tool schema that Claude must call to return structured output ──────────────

_SUBMIT_PLAN_TOOL: dict = {
    "name": "submit_rehab_plan",
    "description": (
        "Submit the complete rehabilitation exercise plan. "
        "You MUST call this tool — do not return plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "exercises": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "name":         {"type": "string"},
                        "sets":         {"type": "integer", "minimum": 1},
                        "reps":         {"type": "string",
                                         "description": "e.g. '10' or '30 seconds'"},
                        "hold_seconds": {"type": "integer", "minimum": 0},
                        "cues":         {"type": "array", "items": {"type": "string"},
                                         "minItems": 1},
                        "rationale":    {"type": "string"},
                        "rag_source_id":{"type": "string",
                                         "description": "chunk_id from the knowledge base"},
                        "rag_excerpt":  {"type": "string",
                                         "description": "verbatim excerpt from that chunk"},
                        "contraindications": {"type": "array",
                                              "items": {"type": "string"}},
                    },
                    "required": [
                        "name", "sets", "reps", "cues",
                        "rationale", "rag_source_id", "rag_excerpt",
                    ],
                },
            },
            "goal_protocol_conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "patient_goal":      {"type": "string"},
                        "protocol_position": {"type": "string"},
                        "resolution":        {"type": "string"},
                    },
                    "required": ["patient_goal", "protocol_position", "resolution"],
                },
            },
            "week_summary":  {
                "type": "string",
                "description": "2-3 sentence patient-facing summary. No banned words.",
            },
            "pt_flag_notes": {
                "type": "string",
                "description": "Clinical concerns for PT. Empty string if none.",
            },
        },
        "required": ["exercises", "goal_protocol_conflicts", "week_summary", "pt_flag_notes"],
    },
}

_SUBMIT_SUMMARY_TOOL: dict = {
    "name": "submit_weekly_summary",
    "description": (
        "Submit the complete weekly rehabilitation progress summary. "
        "You MUST call this tool — do not return plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patient_summary": {
                "type": "string",
                "description": (
                    "Patient-facing. HARD CONSTRAINTS (code-verified after submission): "
                    "(1) under 120 words; "
                    "(2) first sentence = concrete improvement from this week; "
                    "(3) final sentence = one next priority in patient's goal language; "
                    "(4) banned: 'behind', 'should', '!', 'you need to', 'you must'."
                ),
            },
            "next_priority": {
                "type": "string",
                "description": (
                    "One sentence only. Patient-facing. "
                    "Use the patient's own words from their stated goal. "
                    "Same banned-word rules as patient_summary."
                ),
            },
            "graft_citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "chunk_ids from the knowledge base supporting every graft-specific claim "
                    "in patient_summary. Empty list if no graft-specific claims are made. "
                    "Uncited graft claims cause the entire response to be rejected."
                ),
            },
            "pt_bullets": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
                "description": "PT-facing clinical bullets with specific numbers.",
            },
            "adherence_pct": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Exercise adherence rate for this week (0–100).",
            },
            "pt_action_items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific actions for the PT this week. Empty list if none.",
            },
        },
        "required": [
            "patient_summary", "next_priority", "graft_citations",
            "pt_bullets", "adherence_pct", "pt_action_items",
        ],
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    return anthropic.Anthropic(api_key=key)


def _anonymise(patient: PatientProfile) -> dict[str, Any]:
    """Strip all PII before the payload is sent to the API."""
    return {
        "side": patient.side,
        "graft_type": patient.graft_type,
        "weeks_post_op": patient.weeks_post_op,
        "surgery_date_year": patient.surgery_date.year,   # year only, not full date
        "weight_bearing_status": patient.weight_bearing_status,
        "meniscal_repair": patient.meniscal_repair,
        "protocol": patient.protocol,
        "stated_goal_text": patient.stated_goal_text,     # patient's own words — not PII
    }


def _gather_rag_context(patient: PatientProfile, week_start: int) -> tuple[str, list[str]]:
    """
    Run four targeted RAG queries and return (formatted_context, chunk_ids).
    Raises RuntimeError if the index returns no results at all.
    """
    queries = [
        f"{patient.protocol} week {week_start} prescribed exercises",
        f"{patient.graft_type} graft precautions week {week_start}",
        f"meniscal repair {patient.meniscal_repair} exercise restrictions",
        f"{patient.weight_bearing_status} weight bearing strengthening exercises",
    ]

    seen_ids: set[str] = set()
    blocks: list[str] = []

    from rag.retriever import query_with_metadata  # returns (text, chunk_id, score)
    protocol_str = str(patient.protocol)  # ensure plain str, not enum member

    for q in queries:
        # Try protocol-filtered first; fall back to unfiltered if nothing matches.
        # RuntimeError from query_with_metadata (e.g. empty collection) propagates up.
        try:
            results = query_with_metadata(q, protocol=protocol_str, top_k=3)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"RAG query failed: {exc}") from exc
        if not results:
            results = query_with_metadata(q, protocol=None, top_k=3)
        for text, chunk_id, score in results:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                blocks.append(f"[chunk_id: {chunk_id}] (score: {score:.3f})\n{text}")

    if not blocks:
        raise RuntimeError(
            "RAG queries returned no documents even without a protocol filter. "
            "Check that the PDF was ingested and the collection is non-empty."
        )

    return "\n\n---\n\n".join(blocks), list(seen_ids)


def _extract_tool_input(response: anthropic.types.Message, tool_name: str) -> dict:
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            inp = block.input
            # Some SDK versions return block.input as a JSON string rather than a dict.
            if isinstance(inp, str):
                inp = json.loads(inp)
            if not isinstance(inp, dict):
                raise ValueError(
                    f"Tool input for '{tool_name}' is {type(inp).__name__}, expected dict. "
                    f"Raw value: {inp!r}"
                )
            return inp
    raise ValueError(
        f"Model did not call '{tool_name}'. "
        f"stop_reason={response.stop_reason}. "
        f"Content: {response.content}"
    )


def _save_red_flags_for_conflicts(
    patient_id: int,
    conflicts: list[dict],
    pt_flag_notes: str,
) -> None:
    if not conflicts and not pt_flag_notes:
        return
    summaries = [c["resolution"] for c in conflicts]
    if pt_flag_notes:
        summaries.append(pt_flag_notes)
    flag = RedFlagEvent(
        patient_id=patient_id,
        flags=summaries,
        pain_score=0,
        swelling=SwellingLevel.NONE,
        giving_way=False,
    )
    with get_db() as db:
        db.save_red_flag(flag)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_plan(
    patient: PatientProfile,
    consent_record_id: int,
    week_start: int,
    week_end: int,
) -> RehabPlan:
    """
    Generate an evidence-based rehab plan using Claude.

    This is ONE of only TWO functions in this codebase that call the Anthropic API.
    Must be called only after patient has given explicit consent (ConsentRecord saved).

    Raises:
        PermissionError  — consent record not found or revoked.
        RuntimeError     — RAG index empty, or model did not return tool call.
        ValueError       — model returned exercises missing rag_source_id.
    """
    if patient.id is None:
        raise ValueError("Patient must be saved (have an id) before generating a plan.")

    # ── 1. Verify consent ─────────────────────────────────────────────────────
    with get_db() as db:
        consent = db.get_active_consent(patient.id, ConsentType.PLAN_GENERATION)
    if consent is None or consent.id != consent_record_id:
        raise PermissionError(
            "No active consent record found for plan generation. "
            "The patient must consent before a plan can be generated."
        )

    # ── 2. Build anonymised context ───────────────────────────────────────────
    anon = _anonymise(patient)
    rag_context, chunk_ids = _gather_rag_context(patient, week_start)

    user_prompt = PLAN_GENERATION_USER.format(
        protocol=anon["protocol"],
        weeks_post_op=anon["weeks_post_op"],
        week_start=week_start,
        week_end=week_end,
        side=anon["side"],
        graft_type=anon["graft_type"],
        weight_bearing_status=anon["weight_bearing_status"],
        meniscal_repair=anon["meniscal_repair"],
        stated_goal_text=anon["stated_goal_text"],
        rag_context=rag_context,
    )

    # ── 3. Call API ───────────────────────────────────────────────────────────
    client = _build_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=PLAN_GENERATION_SYSTEM,
        tools=[_SUBMIT_PLAN_TOOL],
        tool_choice={"type": "any"},   # force tool use
        messages=[{"role": "user", "content": user_prompt}],
    )

    # ── 4. Parse and validate structured output ───────────────────────────────
    output = _extract_tool_input(response, "submit_rehab_plan")

    raw_exercises = output.get("exercises", [])
    # Defensively parse if Claude returned exercises as a JSON string.
    if isinstance(raw_exercises, str):
        raw_exercises = json.loads(raw_exercises)
    if not isinstance(raw_exercises, list):
        raise ValueError(
            f"'exercises' field is {type(raw_exercises).__name__}, expected list. "
            f"Value: {raw_exercises!r}"
        )
    exercises: list[dict] = []
    for i, item in enumerate(raw_exercises):
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                raise ValueError(
                    f"exercises[{i}] is a plain string, not a dict: {item!r}. "
                    "Model did not follow the tool schema."
                )
        if not isinstance(item, dict):
            raise ValueError(
                f"exercises[{i}] is {type(item).__name__}, expected dict. Value: {item!r}"
            )
        exercises.append(item)

    missing_citations = [
        ex.get("name", f"exercise[{i}]") for i, ex in enumerate(exercises)
        if not ex.get("rag_source_id", "").strip()
    ]
    if missing_citations:
        raise ValueError(
            f"Model returned exercises without RAG citations: {missing_citations}. "
            "Retry or inspect the knowledge base."
        )

    goal_conflicts: list[dict] = output.get("goal_protocol_conflicts", [])
    pt_flag_notes: str = output.get("pt_flag_notes", "")
    week_summary: str = output.get("week_summary", "")

    # ── 5. Flag goal-protocol conflicts for PT review ─────────────────────────
    if goal_conflicts or pt_flag_notes:
        _save_red_flags_for_conflicts(patient.id, goal_conflicts, pt_flag_notes)

    # ── 6. Assemble and return plan (caller saves to DB) ─────────────────────
    cited_ids = list({ex["rag_source_id"] for ex in exercises})
    return RehabPlan(
        patient_id=patient.id,
        consent_record_id=consent_record_id,
        protocol=patient.protocol,
        week_start=week_start,
        week_end=week_end,
        exercises=exercises,
        model_used=MODEL,
        rag_sources=cited_ids,
        week_summary=week_summary,
        pt_flag_notes=pt_flag_notes,
        goal_protocol_conflicts=goal_conflicts,
        review_status=PlanReviewStatus.PENDING,
    )


def generate_weekly_summary(
    patient: PatientProfile,
    consent_record_id: int,
    session_count: int,
    avg_pain: float,
    avg_rpe: float,
    exercises_completed: list[str],
    exercises_skipped: list[str],
    red_flag_count: int,
    recent_notes: str,
) -> dict:
    """
    Generate a patient-facing weekly progress summary using Claude.

    This is ONE of only TWO functions in this codebase that call the Anthropic API.
    Must be called only after patient has given explicit consent (ConsentRecord saved).

    Post-call invariants enforced before returning:
      - patient_summary word count ≤ 120
      - No banned words in patient_summary or next_priority
      - graft_citations non-empty if patient_summary contains graft-specific language

    Raises:
        PermissionError — no active consent record.
        ValueError      — model violated any post-call invariant; caller should retry or flag.
    """
    if patient.id is None:
        raise ValueError("Patient must be saved before generating a summary.")

    # ── 1. Consent gate ───────────────────────────────────────────────────────
    with get_db() as db:
        consent = db.get_active_consent(patient.id, ConsentType.WEEKLY_SUMMARY)
    if consent is None or consent.id != consent_record_id:
        raise PermissionError(
            "No active consent record found for weekly summary generation."
        )

    # ── 2. RAG context for graft-specific claims ──────────────────────────────
    graft_type: str = patient.graft_type
    _GRAFT_QUERIES: dict[str, str] = {
        "hamstring":      f"hamstring graft ACL rehabilitation precautions {patient.protocol}",
        "patellar_tendon": f"patellar tendon graft open chain quad loading restrictions {patient.protocol}",
        "quadriceps":     f"quadriceps tendon graft healing timeline {patient.protocol}",
        "allograft":      f"allograft ACL rehabilitation progression {patient.protocol}",
    }
    graft_rag_context = "No graft-specific excerpts available."
    if graft_type in _GRAFT_QUERIES:
        from rag.retriever import query_with_metadata
        hits = query_with_metadata(_GRAFT_QUERIES[graft_type], protocol=patient.protocol, top_k=3)
        if hits:
            graft_rag_context = "\n\n---\n\n".join(
                f"[chunk_id: {cid}] (score: {score:.3f})\n{text}"
                for text, cid, score in hits
            )

    # ── 3. Build prompt (anonymised — name never included) ────────────────────
    user_prompt = WEEKLY_SUMMARY_USER.format(
        protocol=patient.protocol,
        weeks_post_op=patient.weeks_post_op,
        graft_type=graft_type,
        stated_goal_text=patient.stated_goal_text,
        session_count=session_count,
        avg_pain=f"{avg_pain:.1f}",
        avg_rpe=f"{avg_rpe:.1f}",
        exercises_completed=", ".join(exercises_completed) or "none",
        exercises_skipped=", ".join(exercises_skipped) or "none",
        red_flag_count=red_flag_count,
        recent_notes=recent_notes or "none",
        graft_rag_context=graft_rag_context,
    )

    # ── 4. API call ───────────────────────────────────────────────────────────
    client = _build_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=WEEKLY_SUMMARY_SYSTEM,
        tools=[_SUBMIT_SUMMARY_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    output = _extract_tool_input(response, "submit_weekly_summary")

    # ── 5. Post-call invariant checks (code-level, not prompt-level) ──────────
    _validate_summary_output(output, graft_type)

    return output


# ── Post-call validation helpers ──────────────────────────────────────────────

_GRAFT_KEYWORDS: frozenset[str] = frozenset({
    "graft", "hamstring", "patellar", "quadriceps", "tendon", "allograft",
    "autograft", "harvest", "donor",
})

_NON_TRIVIAL_GRAFTS: frozenset[str] = frozenset({
    "hamstring", "patellar_tendon", "quadriceps", "allograft",
})


def _validate_summary_output(output: dict, graft_type: str) -> None:
    """
    Raise ValueError for any constraint violation.
    Called immediately after the API response — before any DB write or UI display.
    """
    summary: str = output.get("patient_summary", "")
    next_priority: str = output.get("next_priority", "")

    # 1. Word count
    word_count = len(summary.split())
    if word_count > 120:
        raise ValueError(
            f"patient_summary is {word_count} words — limit is 120. "
            "Retry with max_tokens=800 and stricter prompt."
        )

    # 2. Banned words in patient-facing fields
    for field_name, text in [("patient_summary", summary), ("next_priority", next_priority)]:
        hits = check_tone(text)
        if hits:
            raise ValueError(
                f"{field_name} contains banned words {hits}. "
                "Model did not follow tone constraints."
            )

    # 3. Graft-specific claims must be cited
    if graft_type in _NON_TRIVIAL_GRAFTS:
        summary_lower = summary.lower()
        makes_graft_claim = any(kw in summary_lower for kw in _GRAFT_KEYWORDS)
        if makes_graft_claim and not output.get("graft_citations"):
            raise ValueError(
                f"patient_summary references graft keywords but graft_citations is empty. "
                f"Graft type: {graft_type}. Add chunk_id citations from the knowledge base."
            )


# ── Session-coaching tool DEFINITIONS (no API calls here) ────────────────────
# These are passed to state_machine.py which handles its own agentic loop.

SESSION_COACHING_TOOLS: list[dict] = [
    {
        "name": "log_exercise_completion",
        "description": "Record that a patient completed or skipped an exercise.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exercise_name": {"type": "string"},
                "completed":     {"type": "boolean"},
                "pain_during":   {"type": "integer", "minimum": 0, "maximum": 10},
                "notes":         {"type": "string"},
            },
            "required": ["exercise_name", "completed"],
        },
    },
    {
        "name": "flag_for_pt_review",
        "description": (
            "Create a soft alert for the PT. "
            "Non-emergency only — use agent/red_flags.py for emergencies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "detail":  {"type": "string"},
            },
            "required": ["summary", "detail"],
        },
    },
    {
        "name": "rag_query",
        "description": "Search the protocol knowledge base for evidence-based guidance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":    {"type": "string"},
                "protocol": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]
