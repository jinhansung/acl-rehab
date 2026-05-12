"""Unit tests for the deterministic FSM — no DB required (uses in-memory SQLite)."""
from __future__ import annotations

import os
import pytest

os.environ.setdefault("DB_PATH", ":memory:")

from agent.state_machine import (
    InvalidTransitionError,
    RehabState,
    RehabStateMachine,
    STATE_GRAPH,
    ToolNotAllowedError,
)
from data.db import get_db


@pytest.fixture(autouse=True)
def _fresh_db():
    """Each test gets a clean in-memory DB."""
    import data.db as db_module
    db_module.DB_PATH = ":memory:"
    yield
    db_module.DB_PATH = ":memory:"


def _make_patient() -> int:
    """Insert a minimal patient row and return its id."""
    from datetime import date, datetime
    with get_db() as db:
        cur = db._conn.execute(
            """INSERT INTO patients
               (name, side, graft_type, surgery_date, weight_bearing_status,
                meniscal_repair, stated_goal_text, protocol, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("Test", "Left", "hamstring", "2026-01-01", "full",
             "none", "Run again.", "MOON", datetime.utcnow().isoformat()),
        )
        db._conn.commit()
        return cur.lastrowid


# ── Graph completeness ────────────────────────────────────────────────────────

def test_all_states_defined():
    for state in RehabState:
        assert state in STATE_GRAPH, f"{state} missing from STATE_GRAPH"


def test_allowed_next_are_valid_states():
    for state, defn in STATE_GRAPH.items():
        for next_state in defn.allowed_next:
            assert next_state in RehabState, f"Invalid next state: {next_state}"


def test_completed_is_terminal():
    assert not STATE_GRAPH[RehabState.COMPLETED].allowed_next


def test_only_active_has_tools():
    for state, defn in STATE_GRAPH.items():
        if state == RehabState.ACTIVE:
            assert defn.allowed_tools, "ACTIVE must have allowed tools"
        else:
            assert not defn.allowed_tools, f"{state} must have no allowed tools"


# ── Happy-path transitions ────────────────────────────────────────────────────

def test_full_happy_path():
    pid = _make_patient()
    fsm = RehabStateMachine(pid)

    assert fsm.get_state() == RehabState.ONBOARDING

    fsm.trigger_plan_submitted()
    assert fsm.get_state() == RehabState.PLAN_PENDING_PT

    fsm.trigger_plan_approved()
    assert fsm.get_state() == RehabState.ACTIVE

    fsm.trigger_red_flag()
    assert fsm.get_state() == RehabState.RED_FLAG

    fsm.trigger_flag_cleared()
    assert fsm.get_state() == RehabState.ACTIVE

    fsm.trigger_completed()
    assert fsm.get_state() == RehabState.COMPLETED


def test_plan_resubmit_after_rejection_stays_pending():
    """PT rejection does not change state; a new plan re-fires plan_submitted."""
    pid = _make_patient()
    fsm = RehabStateMachine(pid)
    fsm.trigger_plan_submitted()
    assert fsm.get_state() == RehabState.PLAN_PENDING_PT
    # simulate rejection + new plan submitted
    fsm.trigger_plan_submitted()
    assert fsm.get_state() == RehabState.PLAN_PENDING_PT


def test_history_recorded():
    pid = _make_patient()
    fsm = RehabStateMachine(pid)
    fsm.trigger_plan_submitted()
    fsm.trigger_plan_approved()
    history = fsm.get_history()
    assert len(history) == 2
    assert history[0]["from_state"] == "onboarding"
    assert history[0]["to_state"] == "plan_pending_pt"
    assert history[1]["from_state"] == "plan_pending_pt"
    assert history[1]["to_state"] == "active"


# ── Invalid transitions ───────────────────────────────────────────────────────

def test_cannot_approve_from_onboarding():
    pid = _make_patient()
    with pytest.raises(InvalidTransitionError):
        RehabStateMachine(pid).trigger_plan_approved()


def test_cannot_complete_from_red_flag():
    pid = _make_patient()
    fsm = RehabStateMachine(pid)
    fsm.trigger_plan_submitted()
    fsm.trigger_plan_approved()
    fsm.trigger_red_flag()
    with pytest.raises(InvalidTransitionError):
        fsm.trigger_completed()


def test_cannot_red_flag_from_onboarding():
    pid = _make_patient()
    with pytest.raises(InvalidTransitionError):
        RehabStateMachine(pid).trigger_red_flag()


def test_cannot_transition_from_completed():
    pid = _make_patient()
    fsm = RehabStateMachine(pid)
    fsm.trigger_plan_submitted()
    fsm.trigger_plan_approved()
    fsm.trigger_completed()
    with pytest.raises(InvalidTransitionError):
        fsm.trigger_red_flag()


# ── Tool gating ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", [
    "log_exercise_completion",
    "flag_for_pt_review",
    "rag_query",
])
def test_tools_allowed_only_in_active(tool):
    pid = _make_patient()
    fsm = RehabStateMachine(pid)

    # ONBOARDING — blocked
    with pytest.raises(ToolNotAllowedError):
        fsm.assert_tool_allowed(tool)

    fsm.trigger_plan_submitted()
    # PLAN_PENDING_PT — blocked
    with pytest.raises(ToolNotAllowedError):
        fsm.assert_tool_allowed(tool)

    fsm.trigger_plan_approved()
    # ACTIVE — allowed
    fsm.assert_tool_allowed(tool)  # must not raise

    fsm.trigger_red_flag()
    # RED_FLAG — blocked
    with pytest.raises(ToolNotAllowedError):
        fsm.assert_tool_allowed(tool)


def test_unknown_tool_blocked_in_active():
    pid = _make_patient()
    fsm = RehabStateMachine(pid)
    fsm.trigger_plan_submitted()
    fsm.trigger_plan_approved()
    with pytest.raises(ToolNotAllowedError):
        fsm.assert_tool_allowed("generate_plan")   # API call — must never be allowed in FSM
