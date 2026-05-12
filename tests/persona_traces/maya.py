"""
Persona trace — Maya
────────────────────
Maya: data-driven, wants hard numbers and honest timelines.

Clinical context
  Protocol:        MOON
  Weeks post-op:   3
  Side:            Right
  Graft:           Hamstring (eccentrics restricted < 8 weeks)
  Meniscal repair: Medial  (no deep flexion > 90° < 12 weeks)
  WB status:       Full
  Goal (verbatim): "return to volleyball by next season"

Trace steps
  1.  Onboarding  — PatientProfile saved, FSM = ONBOARDING
  2.  Consent     — ConsentRecord saved for plan generation
  3.  Plan gen    — generate_plan() called (API mocked)
  4.  Plan gating — review_status == PENDING; FSM = PLAN_PENDING_PT
  5.  Tool gating — session tools blocked before PT approval
  6.  PT approval — db.approve_plan() + trigger_plan_approved(); FSM = ACTIVE
  7.  Sessions    — 3 × SessionRecord (pain 4/10) logged
  8.  Summary     — generate_weekly_summary() called (API mocked)

Assertions
  A.  Every exercise has a non-empty rag_source_id.
  B.  week_summary contains no banned words.
  C.  Plan review_status == PENDING immediately after generation.
  D.  FSM blocks session tools while PLAN_PENDING_PT.
  E.  FSM reaches ACTIVE only after PT approval.
  F.  Weekly summary patient_summary: ≤ 120 words, no banned words.
  G.  Weekly summary next_priority mirrors volleyball goal language.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-maya")

import data.db as db_module
from agent.prompts import BANNED_WORDS, check_tone
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


# ── Shared mock builders ──────────────────────────────────────────────────────

def _mock_plan_response() -> MagicMock:
    """
    Fake API response for submit_rehab_plan.
    Exercises are safe for hamstring graft wk 3 + medial meniscal repair:
      - no hamstring eccentrics
      - no deep flexion > 90°
    week_summary has no banned words and is patient-facing.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_rehab_plan"
    tool_block.input = {
        "exercises": [
            {
                "name": "Quad Sets",
                "sets": 3,
                "reps": "15",
                "hold_seconds": 5,
                "cues": ["Tighten quad gently", "Keep heel flat"],
                "rationale": "Early quad activation without joint loading",
                "rag_source_id": "moon_wk3_quad_sets_001",
                "rag_excerpt": "Quad sets are recommended in MOON weeks 1–4 for gentle activation.",
                "contraindications": [],
            },
            {
                "name": "Straight Leg Raise",
                "sets": 3,
                "reps": "12",
                "hold_seconds": 0,
                "cues": ["Keep knee straight", "Raise to 45 degrees"],
                "rationale": "Hip flexor and quad co-activation in full extension",
                "rag_source_id": "moon_wk3_slr_002",
                "rag_excerpt": "SLR progresses quad control without knee flexion load.",
                "contraindications": [],
            },
            {
                "name": "Heel Slides",
                "sets": 3,
                "reps": "10",
                "hold_seconds": 0,
                "cues": ["Slide to comfortable range", "Stop at 80 degrees flexion"],
                "rationale": "ROM recovery within medial meniscal repair limits (< 90 degrees)",
                "rag_source_id": "moon_wk3_heel_slides_003",
                "rag_excerpt": "Heel slides to 90 degrees maximum for meniscal repair patients in weeks 1–12.",
                "contraindications": ["Do not force past 90 degrees — medial meniscal repair"],
            },
            {
                "name": "Ankle Pumps",
                "sets": 3,
                "reps": "20",
                "hold_seconds": 0,
                "cues": ["Full dorsiflexion and plantarflexion", "Slow controlled pace"],
                "rationale": "DVT prevention and early circulation support",
                "rag_source_id": "moon_wk3_ankle_pumps_004",
                "rag_excerpt": "Ankle pumps are standard DVT prophylaxis in early ACL rehab.",
                "contraindications": [],
            },
            {
                "name": "Mini Squat (0–45°)",
                "sets": 2,
                "reps": "10",
                "hold_seconds": 0,
                "cues": ["Weight through heel", "Stop at 45 degrees"],
                "rationale": "Closed-chain quad activation within meniscal safe range",
                "rag_source_id": "moon_wk3_mini_squat_005",
                "rag_excerpt": "Mini squats to 45 degrees are permitted with full weight bearing at week 3.",
                "contraindications": ["Limit to 45 degrees — medial meniscal repair"],
            },
        ],
        "goal_protocol_conflicts": [
            {
                "patient_goal": "return to volleyball by next season",
                "protocol_position": "Week 3 MOON protocol focuses on ROM and quad activation only",
                "resolution": "Volleyball-specific agility is appropriate from week 16 onward; PT to counsel on timeline",
            }
        ],
        "week_summary": (
            "This week focuses on rebuilding quad control and recovering knee movement. "
            "Each exercise works within the safe range for your repair. "
            "Your PT will confirm your progress toward volleyball at the next review."
        ),
        "pt_flag_notes": "Medial meniscal repair: maintain < 90° flexion restriction through week 12.",
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _mock_summary_response() -> MagicMock:
    """
    Fake API response for submit_weekly_summary.
    patient_summary: < 120 words, no banned words, first sentence = improvement,
    last sentence mirrors volleyball goal language.
    No graft-specific claims → graft_citations can be empty.
    """
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "submit_weekly_summary"
    tool_block.input = {
        "patient_summary": (
            "Your average pain this week was 4 out of 10 across all three sessions, "
            "showing consistent tolerance for the current exercises. "
            "Quad sets and straight leg raises were completed in every session. "
            "The next step toward returning to volleyball is progressing your knee "
            "range of motion to 90 degrees over the coming week."
        ),
        "next_priority": (
            "Work toward reaching 90 degrees of knee flexion — a key milestone on "
            "the path back to volleyball."
        ),
        "graft_citations": [],
        "pt_bullets": [
            "Week 3 adherence: 3/3 sessions completed (100%).",
            "Average pain NRS 4.0/10; no sessions exceeded 4/10.",
            "No red flags triggered this week.",
            "Medial meniscal repair restriction (< 90° flexion) respected in all sessions.",
            "Goal-protocol conflict logged: volleyball timeline discussed; PT to counsel.",
        ],
        "adherence_pct": 100.0,
        "pt_action_items": [
            "Review ROM progress and confirm < 90° restriction maintained.",
            "Counsel patient on realistic volleyball return timeline (week 16+).",
        ],
    }
    response = MagicMock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


def _fake_rag_results(n: int = 2) -> list[tuple[str, str, float]]:
    return [
        (f"Protocol excerpt {i}", f"chunk_moon_{i:03d}", 0.85 - i * 0.03)
        for i in range(n)
    ]


# ── Trace ─────────────────────────────────────────────────────────────────────

def test_full_trace():
    # ── 1. Onboarding ─────────────────────────────────────────────────────────
    surgery_date = date.today() - timedelta(days=16)   # ~week 3
    with get_db() as db:
        patient = PatientProfile(
            name="Maya",
            side="Right",
            graft_type=GraftType.HAMSTRING,
            surgery_date=surgery_date,
            weight_bearing_status=WeightBearingStatus.FULL,
            meniscal_repair=MeniscalRepair.MEDIAL,
            stated_goal_text="return to volleyball by next season",
            protocol=Protocol.MOON,
        )
        patient_id = db.save_patient(patient)
        patient = patient.model_copy(update={"id": patient_id})

    assert patient_id is not None
    fsm = RehabStateMachine(patient_id)
    assert fsm.get_state() == RehabState.ONBOARDING, "Should start in ONBOARDING"

    # ── 2. Consent ────────────────────────────────────────────────────────────
    payload_hash = ConsentRecord.make_hash("maya-plan-consent")
    with get_db() as db:
        consent = ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.PLAN_GENERATION,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=payload_hash,
        )
        consent_id = db.save_consent(consent)

    assert consent_id is not None

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
            week_start=3,
            week_end=4,
        )

    # ── A: Every exercise cites the protocol knowledge base ───────────────────
    for ex in plan.exercises:
        assert ex.get("rag_source_id", "").strip(), (
            f"Exercise '{ex.get('name')}' has no rag_source_id"
        )

    # ── B: week_summary has no banned words ───────────────────────────────────
    tone_violations = check_tone(plan.week_summary)
    assert not tone_violations, (
        f"week_summary contains banned words {tone_violations}: {plan.week_summary!r}"
    )

    # ── C: Plan is PENDING immediately after generation ───────────────────────
    assert plan.review_status == PlanReviewStatus.PENDING, (
        f"Plan review_status is {plan.review_status!r} — expected 'pending'"
    )

    # Save plan to DB so FSM and tool-gating tests can read it
    with get_db() as db:
        plan_id = db.save_rehab_plan(plan)

    # Advance FSM to PLAN_PENDING_PT
    fsm.trigger_plan_submitted()
    assert fsm.get_state() == RehabState.PLAN_PENDING_PT

    # ── D: Session tools are blocked before PT approval ───────────────────────
    for tool in ("log_exercise_completion", "flag_for_pt_review", "rag_query"):
        with pytest.raises(ToolNotAllowedError):
            fsm.assert_tool_allowed(tool)

    # ── 4. PT approval ────────────────────────────────────────────────────────
    with get_db() as db:
        db.approve_plan(plan_id, pt_notes="Looks appropriate for week 3.")

    fsm.trigger_plan_approved()

    # ── E: FSM reaches ACTIVE only after approval ─────────────────────────────
    assert fsm.get_state() == RehabState.ACTIVE, (
        f"Expected ACTIVE after PT approval, got {fsm.get_state()}"
    )
    # Tools now unblocked
    for tool in ("log_exercise_completion", "flag_for_pt_review", "rag_query"):
        fsm.assert_tool_allowed(tool)   # must not raise

    # Verify DB reflects approved status
    with get_db() as db:
        saved_plan = db.get_plan(plan_id)
    assert saved_plan.review_status == PlanReviewStatus.APPROVED

    # ── 5. Three daily sessions (pain 4/10) ───────────────────────────────────
    exercise_names = [ex["name"] for ex in plan.exercises]
    for session_day_offset in range(3):
        session = SessionRecord(
            patient_id=patient_id,
            date=date.today() - timedelta(days=2 - session_day_offset),
            week_number=3,
            pain_score=4,
            swelling=SwellingLevel.NONE,
            giving_way=False,
            exercises_completed=exercise_names,
            exercises_skipped=[],
            session_notes="Felt manageable. Stopped heel slides at ~80 degrees.",
        )
        with get_db() as db:
            db.save_session(session)

    with get_db() as db:
        sessions = db.get_sessions(patient_id)
    assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}"
    assert all(s.pain_score == 4 for s in sessions)

    # ── 6. Weekly summary (API mocked) ────────────────────────────────────────
    summary_consent_hash = ConsentRecord.make_hash("maya-summary-consent")
    with get_db() as db:
        summary_consent = ConsentRecord(
            patient_id=patient_id,
            consent_type=ConsentType.WEEKLY_SUMMARY,
            model_used="claude-sonnet-4-20250514",
            data_sent_hash=summary_consent_hash,
        )
        summary_consent_id = db.save_consent(summary_consent)

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
            avg_rpe=5.0,
            exercises_completed=exercise_names,
            exercises_skipped=[],
            red_flag_count=0,
            recent_notes="Stopped heel slides at comfortable range.",
        )

    # ── F: patient_summary ≤ 120 words, no banned words ──────────────────────
    patient_summary = summary["patient_summary"]
    word_count = len(patient_summary.split())
    assert word_count <= 120, (
        f"patient_summary is {word_count} words — limit is 120"
    )
    summary_tone = check_tone(patient_summary)
    assert not summary_tone, (
        f"patient_summary has banned words {summary_tone}: {patient_summary!r}"
    )

    # ── G: next_priority mirrors patient's goal language ─────────────────────
    next_priority = summary["next_priority"]
    assert "volleyball" in next_priority.lower(), (
        f"next_priority does not reference 'volleyball' (patient's goal): {next_priority!r}"
    )
    priority_tone = check_tone(next_priority)
    assert not priority_tone, (
        f"next_priority has banned words {priority_tone}: {next_priority!r}"
    )

    # ── PT bullets present and contain data ───────────────────────────────────
    pt_bullets = summary["pt_bullets"]
    assert 3 <= len(pt_bullets) <= 5, (
        f"Expected 3–5 PT bullets, got {len(pt_bullets)}"
    )

    # ── History: onboarding → plan_pending_pt → active ────────────────────────
    history = fsm.get_history()
    state_sequence = [h["to_state"] for h in history]
    assert "plan_pending_pt" in state_sequence
    assert "active" in state_sequence
    assert state_sequence.index("active") > state_sequence.index("plan_pending_pt")
