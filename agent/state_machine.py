"""
Deterministic rehabilitation FSM.

States:   ONBOARDING → PLAN_PENDING_PT → ACTIVE ⇄ RED_FLAG → COMPLETED
                                                ↑___________↓

No state transition happens without an explicit named trigger.
No LLM is involved in any transition decision.
Each state declares exactly which session tools are permitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet

from data.db import get_db


# ── State enum ────────────────────────────────────────────────────────────────

class RehabState(str, Enum):
    ONBOARDING       = "onboarding"
    PLAN_PENDING_PT  = "plan_pending_pt"
    ACTIVE           = "active"
    RED_FLAG         = "red_flag"
    COMPLETED        = "completed"


# ── State graph definition ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateDefinition:
    description: str
    allowed_tools: FrozenSet[str]
    allowed_next: FrozenSet[RehabState]


STATE_GRAPH: dict[RehabState, StateDefinition] = {
    RehabState.ONBOARDING: StateDefinition(
        description=(
            "Patient profile created. Awaiting consent and AI plan generation."
        ),
        allowed_tools=frozenset(),
        allowed_next=frozenset({RehabState.PLAN_PENDING_PT}),
    ),
    RehabState.PLAN_PENDING_PT: StateDefinition(
        description=(
            "Plan generated and submitted. Awaiting PT approval before sessions begin."
        ),
        allowed_tools=frozenset(),
        allowed_next=frozenset({RehabState.ACTIVE}),
        # NOTE: plan rejection does NOT change state — a new plan must be generated
        # and submitted, which re-uses the same trigger (plan_submitted).
    ),
    RehabState.ACTIVE: StateDefinition(
        description=(
            "Plan approved. Daily sessions permitted. "
            "Red-flag detection runs before every session."
        ),
        allowed_tools=frozenset({
            "log_exercise_completion",
            "flag_for_pt_review",
            "rag_query",
        }),
        allowed_next=frozenset({RehabState.RED_FLAG, RehabState.COMPLETED}),
    ),
    RehabState.RED_FLAG: StateDefinition(
        description=(
            "One or more red flags triggered. No session activities permitted "
            "until PT marks the flag as reviewed."
        ),
        allowed_tools=frozenset(),
        allowed_next=frozenset({RehabState.ACTIVE}),
    ),
    RehabState.COMPLETED: StateDefinition(
        description="Rehabilitation protocol completed. Terminal state.",
        allowed_tools=frozenset(),
        allowed_next=frozenset(),   # terminal — no outgoing transitions
    ),
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class InvalidTransitionError(Exception):
    """Raised when a trigger is fired from an incompatible source state."""


class ToolNotAllowedError(Exception):
    """Raised when a tool is called in a state that does not permit it."""


# ── State machine ─────────────────────────────────────────────────────────────

class RehabStateMachine:
    """
    Thin wrapper around the patient_states DB table.
    All state reads/writes go through the DB — no in-memory state.
    Instantiate per request; do not cache across Streamlit reruns.
    """

    def __init__(self, patient_id: int) -> None:
        self.patient_id = patient_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_state(self) -> RehabState:
        with get_db() as db:
            raw = db.get_patient_state(self.patient_id)
        return RehabState(raw)

    def definition(self) -> StateDefinition:
        return STATE_GRAPH[self.get_state()]

    def tool_allowed(self, tool_name: str) -> bool:
        return tool_name in self.definition().allowed_tools

    def assert_tool_allowed(self, tool_name: str) -> None:
        if not self.tool_allowed(tool_name):
            state = self.get_state()
            allowed = sorted(self.definition().allowed_tools)
            raise ToolNotAllowedError(
                f"Tool '{tool_name}' is not permitted in state '{state}'. "
                f"Allowed tools: {allowed or 'none'}"
            )

    def get_history(self) -> list[dict]:
        with get_db() as db:
            return db.get_state_history(self.patient_id)

    # ── Transitions (one method per valid trigger) ────────────────────────────

    def trigger_plan_submitted(self) -> None:
        """
        ONBOARDING → PLAN_PENDING_PT
        Fire after: patient profile saved + consent recorded + plan generated and saved.
        Also valid from PLAN_PENDING_PT when a replacement plan is submitted after rejection.
        """
        current = self.get_state()
        if current not in (RehabState.ONBOARDING, RehabState.PLAN_PENDING_PT):
            raise InvalidTransitionError(
                f"trigger_plan_submitted requires state ONBOARDING or PLAN_PENDING_PT, "
                f"got '{current}'."
            )
        self._write(RehabState.PLAN_PENDING_PT, "plan_submitted")

    def trigger_plan_approved(self) -> None:
        """
        PLAN_PENDING_PT → ACTIVE
        Fire after: PT approves the plan in the dashboard.
        """
        self._require_source(RehabState.PLAN_PENDING_PT, "trigger_plan_approved")
        self._write(RehabState.ACTIVE, "plan_approved")

    def trigger_red_flag(self) -> None:
        """
        ACTIVE → RED_FLAG
        Fire after: agent/red_flags.check_red_flags() returns a non-empty list.
        Deterministic — no LLM involved.
        """
        self._require_source(RehabState.ACTIVE, "trigger_red_flag")
        self._write(RehabState.RED_FLAG, "red_flag_triggered")

    def trigger_flag_cleared(self) -> None:
        """
        RED_FLAG → ACTIVE
        Fire after: PT marks the open RedFlagEvent as reviewed in the dashboard.
        """
        self._require_source(RehabState.RED_FLAG, "trigger_flag_cleared")
        self._write(RehabState.ACTIVE, "flag_cleared_by_pt")

    def trigger_completed(self) -> None:
        """
        ACTIVE → COMPLETED
        Fire after: PT explicitly marks the patient as having completed the protocol,
        or automatically when the protocol's final week is confirmed complete.
        """
        self._require_source(RehabState.ACTIVE, "trigger_completed")
        self._write(RehabState.COMPLETED, "protocol_completed")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _require_source(self, expected: RehabState, trigger_name: str) -> None:
        current = self.get_state()
        if current != expected:
            raise InvalidTransitionError(
                f"{trigger_name} requires state '{expected}', got '{current}'."
            )

    def _write(self, to_state: RehabState, trigger: str) -> None:
        target_def = STATE_GRAPH[to_state]
        current = self.get_state()
        current_def = STATE_GRAPH[current]
        if to_state not in current_def.allowed_next:
            raise InvalidTransitionError(
                f"Transition '{current}' → '{to_state}' is not in the state graph."
            )
        with get_db() as db:
            db.set_patient_state(self.patient_id, to_state.value, trigger)
