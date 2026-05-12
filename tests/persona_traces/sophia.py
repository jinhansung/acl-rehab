"""
Persona trace — Sophia
──────────────────────
Sophia: teenage patient. Age-appropriate language. No comparisons to other patients
or population norms. Private journal respected. Tiered parent view (not tested here
as it is a UI concern, not a business-logic concern).

Clinical context
  Protocol:        Aspetar
  Weeks post-op:   8
  Side:            Right
  Graft:           Hamstring (eccentrics restricted < 8 weeks — exactly at boundary;
                   plan should note this transition point)
  Meniscal repair: Lateral  (no deep flexion > 90° < 12 weeks)
  WB status:       Full
  Goal (verbatim): "get back to school sports this term"

Trace steps
  1.  Onboarding  — PatientProfile saved, FSM = ONBOARDING
  2.  Consent     — ConsentRecord saved for plan generation
  3.  Plan gen    — generate_plan() called (API mocked)
  4.  Plan gating — review_status == PENDING; FSM = PLAN_PENDING_PT
  5.  Tool gating — session tools blocked before PT approval
  6.  PT approval — db.approve_plan() + trigger_plan_approved(); FSM = ACTIVE
  7.  Sessions    — 3 × SessionRecord (pain 4/10) logged
  8.  Summary     — generate_weekly_summary() called (API mocked)

Assertions (same contract as Maya/Helen, plus Sophia-specific)
  A.  Every exercise has a non-empty rag_source_id.
  B.  week_summary contains no banned words.
  C.  Plan review_status == PENDING immediately after generation.
  D.  FSM blocks session tools while PLAN_PENDING_PT.
  E.  FSM reaches ACTIVE only after PT approval.
  F.  Weekly summary patient_summary: ≤ 120 words, no banned words.
  G.  Weekly summary next_priority mirrors Sophia's goal language (school sports).
  I.  patient_summary makes no comparison to other patients ("others", "average",
      "most patients", "typical").
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-sophia")

import data.db as db_module
from agent.prompts import check_tone
from agent.state_machine import RehabState, RehabStateMachine, ToolNotAllowedError
from agent.tools import generate_plan, generate_weekly_summary
from data.db import get_db
from data.models import (
    ConsentRecord,
    ConsentType,
    GraftType,
    MeniscalRepair,
    PatientProfile,
    PlanReviewStatus,
    Protocol,
    SessionRecord,
    SwellingLevel,
    WeightBearingStatus,
)

# Phrases that would compare Sophia against other patients
_COMPARISON_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bother patients?\b", re.I),
    re.compile(r"\bmost patients?\b", re.I),
    re.compile(r"\btypical(ly)?\b", re.I),
    re.compile(r"\baverage patient\b", re.I),
    re.compile(r"\bnorm(ally)?\b", re.I),
    re.compile(r"\bcompared? to\b", re.I),
]


def _no_patient_comparisons(text: str) -> list[str]:
    """Return list of matched comparison phrases found in text."""
    return [pat.pattern for pat in _COMPARISON_PATTERNS if pat.search(text)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    db_module.DB_PATH = ":memory:"
    yield
    db_module.DB_PATH = ":memory:"


# ── Mock builders ─────────────────────────────────────────────────────────────

def _mock_plan_response() -> MagicMock:
    """
    Fake response for submit_rehab_plan.
    Week 8 hamstring graft + lateral meniscal repair:
      - Hamstring eccentrics now permitted (week 8 = end of restriction window)
        — plan notes this and starts carefully
      - No deep flexion > 90° (lateral meniscal repair through week 12)
    week_summary: age-appropriate, no banned words.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_rehab_plan"
    tool_block.input = {
        "exercises": [
            {
                "name": "Nordic Curl Introduction (Partial)",
                "sets": 2,
                "reps": "6",
                "hold_seconds": 0,
                "cues": ["Start with small range only", "Use a partner or strap"],
                "rationale": (
                    "Week 8 marks the start of the hamstring eccentric window per Aspetar; "
                    "begin with partial range only"
                ),
                "rag_source_id": "aspetar_wk8_nordic_001",
                "rag_excerpt": "Aspetar protocol introduces eccentric hamstring loading at week 8.",
                "contraindications": ["Do not progress past 30° range this week"],
            },
            {
                "name": "Bulgarian Split Squat",
                "sets": 3,
                "reps": "10",
                "hold_seconds": 0,
                "cues": ["Rear foot on bench", "Front knee stays behind toes"],
                "rationale": "Single-leg quad and glute strengthening; closed-chain",
                "rag_source_id": "aspetar_wk8_split_squat_002",
                "rag_excerpt": "Split squat progressions begin at week 8 in Aspetar.",
                "contraindications": ["Keep knee flexion to 90 degrees — lateral meniscal repair"],
            },
            {
                "name": "Leg Press (0–90°)",
                "sets": 3,
                "reps": "12",
                "hold_seconds": 0,
                "cues": ["Stop at 90 degrees", "Push through full foot"],
                "rationale": "Bilateral quad strengthening within safe flexion range",
                "rag_source_id": "aspetar_wk8_leg_press_003",
                "rag_excerpt": "Leg press to 90 degrees is standard at week 8 Aspetar progression.",
                "contraindications": ["Limit to 90 degrees — lateral meniscal repair"],
            },
            {
                "name": "Side Hops (Low Amplitude)",
                "sets": 3,
                "reps": "10 per side",
                "hold_seconds": 0,
                "cues": ["Soft landing", "Keep hops small"],
                "rationale": "Early plyometric introduction for return-to-sport preparation",
                "rag_source_id": "aspetar_wk8_side_hops_004",
                "rag_excerpt": "Low-amplitude lateral hops are introduced at week 8 in Aspetar.",
                "contraindications": [],
            },
            {
                "name": "Calf Raises — Single-Leg",
                "sets": 3,
                "reps": "15",
                "hold_seconds": 2,
                "cues": ["Full range of motion", "Slow controlled descent"],
                "rationale": "Calf and Achilles strengthening for sport readiness",
                "rag_source_id": "aspetar_wk8_calf_raise_005",
                "rag_excerpt": "Single-leg calf raises are a standard component from week 6 Aspetar.",
                "contraindications": [],
            },
        ],
        "goal_protocol_conflicts": [],
        "week_summary": (
            "Week 8 is a big step — your exercises now include more sport-like movements. "
            "The new exercises are designed to get your knee ready for the demands of school sports. "
            "Stay within the ranges shown; your PT will progress things further each week."
        ),
        "pt_flag_notes": (
            "Lateral meniscal repair: maintain < 90° restriction through week 12. "
            "Nordic curl introduced at low range — monitor for posterior knee pain."
        ),
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _mock_summary_response() -> MagicMock:
    """
    Fake response for submit_weekly_summary.
    patient_summary: < 120 words, no banned words, no patient comparisons,
    last sentence mirrors school-sports goal.
    No graft-specific language in patient_summary → graft_citations empty.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_weekly_summary"
    tool_block.input = {
        "patient_summary": (
            "Your pain stayed at 4 out of 10 across all three sessions this week — "
            "a consistent result that shows your knee is tolerating the week 8 exercises well. "
            "Side hops and split squats were completed in every session. "
            "The next step toward getting back to school sports is building up the "
            "number of hops and adding more strength work."
        ),
        "next_priority": (
            "Build up your side hops gradually each session — "
            "that is the key movement for getting back to school sports."
        ),
        "graft_citations": [],
        "pt_bullets": [
            "Week 8 adherence: 3/3 sessions (100%).",
            "Average pain NRS 4.0/10; no sessions exceeded 4/10.",
            "No red flags triggered.",
            "Nordic curl introduced at partial range — no posterior knee pain reported.",
            "Lateral meniscal repair restriction (< 90° flexion) maintained in all sessions.",
        ],
        "adherence_pct": 100.0,
        "pt_action_items": [
            "Review lateral meniscal repair status before progressing flexion range.",
            "Assess Nordic curl tolerance and consider range progression next session.",
        ],
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _fake_rag_results(n: int = 2) -> list[tuple[str, str, float]]:
    return [
        (f"Aspetar excerpt {i}", f"chunk_aspetar_{i:03d}", 0.86 - i * 0.03)
        for i in range(n)
    ]


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_full_trace():
    # ── 1. Onboarding ─────────────────────────────────────────────────────────
    surgery_date = date.today() - timedelta(days=51)   # ~week 8
    with get_db() as db:
        patient = PatientProfile(
            name="Sophia",
            side="Right",
            graft_type=GraftType.HAMSTRING,
            surgery_date=surgery_date,
            weight_bearing_status=WeightBearingStatus.FULL,
            meniscal_repair=MeniscalRepair.LATERAL,
            stated_goal_text="get back to school sports this term",
            protocol=Protocol.ASPETAR,
        )
        patient_id = db.save_patient(patient)
        patient = patient.model_copy(update={"id": patient_id})

    assert patient.weeks_post_op == 8, (
        f"Expected week 8 post-op, got week {patient.weeks_post_op}"
    )

    fsm = RehabStateMachine(patient_id)
    assert fsm.get_state() == RehabState.ONBOARDING

    # ── 2. Consent ────────────────────────────────────────────────────────────
    with get_db() as db:
        consent_id = db.save_consent(ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.PLAN_GENERATION,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=ConsentRecord.make_hash("sophia-plan-consent"),
        ))

    # ── 3. Plan generation (API mocked) ───────────────────────────────────────
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _mock_plan_response()

    with (
        patch("agent.tools._build_client", return_value=mock_client),
        patch("rag.retriever.query_with_metadata", return_value=_fake_rag_results()),
    ):
        plan = generate_plan(
            patient=patient,
            consent_record_id=consent_id,
            week_start=8,
            week_end=9,
        )

    # ── A: Every exercise cites protocol knowledge base ───────────────────────
    for ex in plan.exercises:
        assert ex.get("rag_source_id", "").strip(), (
            f"Exercise '{ex.get('name')}' missing rag_source_id"
        )

    # ── B: week_summary has no banned words ───────────────────────────────────
    assert not check_tone(plan.week_summary), (
        f"week_summary has banned words: {plan.week_summary!r}"
    )

    # ── C: Plan is PENDING immediately after generation ───────────────────────
    assert plan.review_status == PlanReviewStatus.PENDING

    with get_db() as db:
        plan_id = db.save_rehab_plan(plan)

    fsm.trigger_plan_submitted()
    assert fsm.get_state() == RehabState.PLAN_PENDING_PT

    # ── D: Tool gating blocks session tools ───────────────────────────────────
    for tool in ("log_exercise_completion", "flag_for_pt_review", "rag_query"):
        with pytest.raises(ToolNotAllowedError):
            fsm.assert_tool_allowed(tool)

    # ── 4. PT approval ────────────────────────────────────────────────────────
    with get_db() as db:
        db.approve_plan(
            plan_id,
            pt_notes=(
                "Week 8 plan approved. Nordic curl at partial range only. "
                "Review lateral meniscal flexion restriction at week 12."
            ),
        )

    fsm.trigger_plan_approved()

    # ── E: ACTIVE only after PT approval ─────────────────────────────────────
    assert fsm.get_state() == RehabState.ACTIVE
    for tool in ("log_exercise_completion", "flag_for_pt_review", "rag_query"):
        fsm.assert_tool_allowed(tool)

    with get_db() as db:
        assert db.get_plan(plan_id).review_status == PlanReviewStatus.APPROVED

    # ── 5. Three daily sessions (pain 4/10) ───────────────────────────────────
    exercise_names = [ex["name"] for ex in plan.exercises]
    for offset in range(3):
        with get_db() as db:
            db.save_session(SessionRecord(
                patient_id=patient_id,
                date=date.today() - timedelta(days=2 - offset),
                week_number=8,
                pain_score=4,
                swelling=SwellingLevel.NONE,
                giving_way=False,
                exercises_completed=exercise_names,
                exercises_skipped=[],
                session_notes=(
                    "Nordic curls kept to small range. "
                    "Knee felt okay during side hops."
                ),
            ))

    with get_db() as db:
        sessions = db.get_sessions(patient_id)
    assert len(sessions) == 3
    assert all(s.pain_score == 4 for s in sessions)

    # ── 6. Weekly summary (API mocked) ────────────────────────────────────────
    with get_db() as db:
        summary_consent_id = db.save_consent(ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.WEEKLY_SUMMARY,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=ConsentRecord.make_hash("sophia-summary-consent"),
        ))

    mock_summary_client = MagicMock()
    mock_summary_client.messages.create.return_value = _mock_summary_response()

    with (
        patch("agent.tools._build_client", return_value=mock_summary_client),
        patch("rag.retriever.query_with_metadata", return_value=_fake_rag_results()),
    ):
        summary = generate_weekly_summary(
            patient=patient,
            consent_record_id=summary_consent_id,
            session_count=3,
            avg_pain=4.0,
            avg_rpe=5.5,
            exercises_completed=exercise_names,
            exercises_skipped=[],
            red_flag_count=0,
            recent_notes="Nordic curls kept small. No posterior knee pain reported.",
        )

    # ── F: patient_summary ≤ 120 words, no banned words ──────────────────────
    patient_summary = summary["patient_summary"]
    assert len(patient_summary.split()) <= 120
    assert not check_tone(patient_summary), (
        f"patient_summary has banned words: {patient_summary!r}"
    )

    # ── G: next_priority mirrors Sophia's goal language ───────────────────────
    next_priority = summary["next_priority"]
    assert any(
        kw in next_priority.lower()
        for kw in ("school", "sport", "back")
    ), (
        f"next_priority does not reflect Sophia's goal: {next_priority!r}"
    )
    assert not check_tone(next_priority)

    # ── I: No comparison to other patients ───────────────────────────────────
    comparisons_in_summary = _no_patient_comparisons(patient_summary)
    assert not comparisons_in_summary, (
        f"patient_summary compares Sophia to other patients "
        f"(patterns: {comparisons_in_summary}): {patient_summary!r}"
    )
    comparisons_in_priority = _no_patient_comparisons(next_priority)
    assert not comparisons_in_priority, (
        f"next_priority compares Sophia to other patients "
        f"(patterns: {comparisons_in_priority}): {next_priority!r}"
    )

    # ── State history ─────────────────────────────────────────────────────────
    history = fsm.get_history()
    to_states = [h["to_state"] for h in history]
    assert to_states == ["plan_pending_pt", "active"]
