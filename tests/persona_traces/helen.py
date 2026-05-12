"""
Persona trace — Helen
─────────────────────
Helen: values flexibility, acknowledges hard days, splits sessions to fit family life.
No streaks mentioned. Split sessions treated as valid, not failures.

Clinical context
  Protocol:        Delaware-Oslo
  Weeks post-op:   6
  Side:            Left
  Graft:           Patellar tendon (aggressive open-chain quad loading restricted < 6 weeks;
                   week 6 = exactly at the transition point — proceed cautiously)
  Meniscal repair: None
  WB status:       Full
  Goal (verbatim): "manage split sessions around my family schedule"

Trace steps
  1.  Onboarding  — PatientProfile saved, FSM = ONBOARDING
  2.  Consent     — ConsentRecord saved for plan generation
  3.  Plan gen    — generate_plan() called (API mocked)
  4.  Plan gating — review_status == PENDING; FSM = PLAN_PENDING_PT
  5.  Tool gating — session tools blocked before PT approval
  6.  PT approval — db.approve_plan() + trigger_plan_approved(); FSM = ACTIVE
  7.  Sessions    — 3 × SessionRecord (pain 4/10) logged; one session split across two records
  8.  Summary     — generate_weekly_summary() called (API mocked)

Assertions (same contract as Maya, plus Helen-specific)
  A.  Every exercise has a non-empty rag_source_id.
  B.  week_summary contains no banned words.
  C.  Plan review_status == PENDING immediately after generation.
  D.  FSM blocks session tools while PLAN_PENDING_PT.
  E.  FSM reaches ACTIVE only after PT approval.
  F.  Weekly summary patient_summary: ≤ 120 words, no banned words.
  G.  Weekly summary next_priority mirrors Helen's goal language (split sessions / family).
  H.  week_summary does not contain the word "streak" (Helen persona rule).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-helen")

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
    Week 6 patellar tendon: open-chain quad loading now permitted with care.
    No meniscal restrictions.
    week_summary: no banned words; acknowledges split sessions as valid.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_rehab_plan"
    tool_block.input = {
        "exercises": [
            {
                "name": "Terminal Knee Extension",
                "sets": 3,
                "reps": "15",
                "hold_seconds": 2,
                "cues": ["Band behind knee", "Push knee straight against band"],
                "rationale": "Open-chain quad activation entering week 6 patellar tendon window",
                "rag_source_id": "delaware_wk6_tke_001",
                "rag_excerpt": "TKE is introduced at week 6 for patellar tendon grafts in Delaware-Oslo.",
                "contraindications": [],
            },
            {
                "name": "Step-Ups",
                "sets": 3,
                "reps": "12",
                "hold_seconds": 0,
                "cues": ["Lead with operated leg", "Control the descent"],
                "rationale": "Closed-chain progression building single-leg strength",
                "rag_source_id": "delaware_wk6_stepup_002",
                "rag_excerpt": "Step-ups are a key closed-chain exercise introduced at week 6.",
                "contraindications": [],
            },
            {
                "name": "Stationary Bike",
                "sets": 1,
                "reps": "15 minutes",
                "hold_seconds": 0,
                "cues": ["Low resistance", "Full pedal stroke"],
                "rationale": "Aerobic conditioning and ROM maintenance",
                "rag_source_id": "delaware_wk6_bike_003",
                "rag_excerpt": "Stationary cycling at low resistance from week 4 onward per Delaware-Oslo.",
                "contraindications": [],
            },
            {
                "name": "Single-Leg Balance",
                "sets": 3,
                "reps": "30 seconds",
                "hold_seconds": 0,
                "cues": ["Slight knee bend", "Focus on a fixed point"],
                "rationale": "Proprioception and neuromuscular re-education",
                "rag_source_id": "delaware_wk6_balance_004",
                "rag_excerpt": "Single-leg balance tasks begin at week 6 in Delaware-Oslo protocol.",
                "contraindications": [],
            },
            {
                "name": "Hip Abduction Side-Lying",
                "sets": 3,
                "reps": "15",
                "hold_seconds": 0,
                "cues": ["Keep hips stacked", "Controlled return"],
                "rationale": "Hip and glute strengthening to reduce knee valgus",
                "rag_source_id": "delaware_wk6_hip_abd_005",
                "rag_excerpt": "Hip abduction exercises support frontal-plane stability throughout rehab.",
                "contraindications": [],
            },
        ],
        "goal_protocol_conflicts": [],
        "week_summary": (
            "This week introduces more active strengthening for your knee. "
            "Each exercise can be split across the day — morning and evening sets "
            "count just as much as a single block. "
            "Progress at a pace that works around your family schedule."
        ),
        "pt_flag_notes": "",
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _mock_summary_response() -> MagicMock:
    """
    Fake response for submit_weekly_summary.
    patient_summary: < 120 words, no banned words, acknowledges split sessions,
    no mention of streaks, last sentence mirrors family-schedule goal.
    No graft-specific claims → graft_citations empty.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_weekly_summary"
    tool_block.input = {
        "patient_summary": (
            "All three sessions were completed this week, with average pain sitting "
            "at 4 out of 10 — a manageable level for the exercises introduced at week 6. "
            "Step-ups and terminal knee extensions were completed in every session. "
            "Continuing to fit sessions around your family schedule is a valid and "
            "effective approach to your rehabilitation."
        ),
        "next_priority": (
            "Keep building the habit of fitting sessions around your family schedule "
            "— any combination of morning and evening sets counts toward your progress."
        ),
        "graft_citations": [],
        "pt_bullets": [
            "Week 6 adherence: 3/3 sessions completed (100%).",
            "Average pain NRS 4.0/10 — within expected range for week 6 loading.",
            "No red flags triggered.",
            "Open-chain quad loading introduced cautiously per patellar tendon graft protocol.",
        ],
        "adherence_pct": 100.0,
        "pt_action_items": [],
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _fake_rag_results(n: int = 2) -> list[tuple[str, str, float]]:
    return [
        (f"Delaware-Oslo excerpt {i}", f"chunk_delaware_{i:03d}", 0.84 - i * 0.03)
        for i in range(n)
    ]


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_full_trace():
    # ── 1. Onboarding ─────────────────────────────────────────────────────────
    surgery_date = date.today() - timedelta(days=37)   # ~week 6
    with get_db() as db:
        patient = PatientProfile(
            name="Helen",
            side="Left",
            graft_type=GraftType.PATELLAR_TENDON,
            surgery_date=surgery_date,
            weight_bearing_status=WeightBearingStatus.FULL,
            meniscal_repair=MeniscalRepair.NONE,
            stated_goal_text="manage split sessions around my family schedule",
            protocol=Protocol.DELAWARE_OSLO,
        )
        patient_id = db.save_patient(patient)
        patient = patient.model_copy(update={"id": patient_id})

    fsm = RehabStateMachine(patient_id)
    assert fsm.get_state() == RehabState.ONBOARDING

    # ── 2. Consent ────────────────────────────────────────────────────────────
    with get_db() as db:
        consent = ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.PLAN_GENERATION,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=ConsentRecord.make_hash("helen-plan-consent"),
        )
        consent_id = db.save_consent(consent)

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
            week_start=6,
            week_end=7,
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

    # ── H: week_summary does not mention "streak" ─────────────────────────────
    assert "streak" not in plan.week_summary.lower(), (
        f"week_summary mentions 'streak' — violates Helen persona rule: {plan.week_summary!r}"
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
        db.approve_plan(plan_id, pt_notes="Week 6 plan appropriate for patellar tendon.")

    fsm.trigger_plan_approved()

    # ── E: ACTIVE only after PT approval ─────────────────────────────────────
    assert fsm.get_state() == RehabState.ACTIVE
    for tool in ("log_exercise_completion", "flag_for_pt_review", "rag_query"):
        fsm.assert_tool_allowed(tool)

    with get_db() as db:
        assert db.get_plan(plan_id).review_status == PlanReviewStatus.APPROVED

    # ── 5. Three sessions — including a split-session pair ────────────────────
    exercise_names = [ex["name"] for ex in plan.exercises]

    # Session 1: full session
    with get_db() as db:
        db.save_session(SessionRecord(
            patient_id=patient_id,
            date=date.today() - timedelta(days=2),
            week_number=6,
            pain_score=4,
            swelling=SwellingLevel.MILD,
            giving_way=False,
            exercises_completed=exercise_names,
            exercises_skipped=[],
            session_notes="Completed in one block, morning.",
        ))

    # Sessions 2a + 2b: split across two records (Helen does this)
    # Both count — neither is penalised
    split_am = exercise_names[:3]
    split_pm = exercise_names[3:]
    with get_db() as db:
        db.save_session(SessionRecord(
            patient_id=patient_id,
            date=date.today() - timedelta(days=1),
            week_number=6,
            pain_score=4,
            swelling=SwellingLevel.NONE,
            giving_way=False,
            exercises_completed=split_am,
            exercises_skipped=[],
            session_notes="Morning block — 3 exercises before school run.",
        ))
        db.save_session(SessionRecord(
            patient_id=patient_id,
            date=date.today() - timedelta(days=1),
            week_number=6,
            pain_score=4,
            swelling=SwellingLevel.NONE,
            giving_way=False,
            exercises_completed=split_pm,
            exercises_skipped=[],
            session_notes="Evening block — remaining exercises after dinner.",
        ))

    with get_db() as db:
        sessions = db.get_sessions(patient_id)

    # 3 records total (1 full + 2 split halves)
    assert len(sessions) == 3
    assert all(s.pain_score == 4 for s in sessions)

    # ── 6. Weekly summary (API mocked) ────────────────────────────────────────
    with get_db() as db:
        summary_consent_id = db.save_consent(ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.WEEKLY_SUMMARY,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=ConsentRecord.make_hash("helen-summary-consent"),
        ))

    mock_summary_client = MagicMock()
    mock_summary_client.messages.create.return_value = _mock_summary_response()

    all_completed = [ex for s in sessions for ex in s.exercises_completed]
    with (
        patch("agent.tools._build_client", return_value=mock_summary_client),
        patch("rag.retriever.query_with_metadata", return_value=_fake_rag_results()),
    ):
        summary = generate_weekly_summary(
            patient=patient,
            consent_record_id=summary_consent_id,
            session_count=3,
            avg_pain=4.0,
            avg_rpe=4.5,
            exercises_completed=list(set(all_completed)),
            exercises_skipped=[],
            red_flag_count=0,
            recent_notes="Split sessions across morning and evening.",
        )

    # ── F: patient_summary ≤ 120 words, no banned words ──────────────────────
    patient_summary = summary["patient_summary"]
    assert len(patient_summary.split()) <= 120
    assert not check_tone(patient_summary), (
        f"patient_summary has banned words: {patient_summary!r}"
    )

    # ── G: next_priority mirrors Helen's goal language ────────────────────────
    next_priority = summary["next_priority"]
    assert any(
        kw in next_priority.lower()
        for kw in ("session", "family", "schedule", "split")
    ), (
        f"next_priority does not reference Helen's goal (split sessions/family): "
        f"{next_priority!r}"
    )
    assert not check_tone(next_priority)

    # ── H (summary): no "streak" in patient-facing fields ────────────────────
    assert "streak" not in patient_summary.lower(), (
        "patient_summary mentions 'streak' — violates Helen persona rule"
    )
    assert "streak" not in next_priority.lower(), (
        "next_priority mentions 'streak' — violates Helen persona rule"
    )

    # ── State history ─────────────────────────────────────────────────────────
    history = fsm.get_history()
    to_states = [h["to_state"] for h in history]
    assert to_states == ["plan_pending_pt", "active"]
