"""
PT dashboard — plan approval queue, red-flag queue, patient roster.

Access is gated by PT code. Three tabs:
  Plan queue  — pending AI-generated plans: approve / edit+approve / reject
  Red flags   — open RedFlagEvents: review / escalate
  Patients    — full roster with protocol override
"""
from __future__ import annotations

import json

import streamlit as st

from data.db import get_db
from data.models import PatientProfile, Protocol, RehabPlan

st.set_page_config(page_title="PT Dashboard", page_icon="🩺", layout="wide")
st.title("PT Dashboard")

# ── Auth gate ─────────────────────────────────────────────────────────────────
PT_CODES: set[str] = {"demo123"}  # replace with real auth

if "pt_authed" not in st.session_state:
    st.session_state.pt_authed = False

if not st.session_state.pt_authed:
    code = st.text_input("PT access code", type="password")
    if st.button("Sign in"):
        if code in PT_CODES:
            st.session_state.pt_authed = True
            st.rerun()
        else:
            st.error("Access code not recognised.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with get_db() as db:
    pending_plans = db.get_pending_plans()   # list[(RehabPlan, PatientProfile)]
    open_flags    = db.get_open_red_flags()
    all_patients  = db.get_all_patients()


# ── Plan review renderer (defined before tabs so tabs can call it) ────────────

def _render_plan_review(plan: RehabPlan, patient: PatientProfile) -> None:
    # Patient context strip
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Graft", patient.graft_type.replace("_", " ").title())
    col_b.metric("Weight bearing", patient.weight_bearing_status.replace("_", " ").title())
    col_c.metric("Meniscal repair", patient.meniscal_repair.title())
    col_d.metric("Weeks post-op", patient.weeks_post_op)

    st.write(f"**Patient goal:** {patient.stated_goal_text}")

    # Goal–protocol conflicts
    if plan.goal_protocol_conflicts:
        st.warning("Goal–protocol conflicts flagged by the model:")
        for conflict in plan.goal_protocol_conflicts:
            st.write(f"- **Goal:** {conflict['patient_goal']}")
            st.write(f"  **Protocol position:** {conflict['protocol_position']}")
            st.write(f"  **Resolution:** {conflict['resolution']}")

    if plan.pt_flag_notes:
        st.info(f"Model notes for PT: {plan.pt_flag_notes}")

    # Week summary — editable before approval
    st.subheader("Week summary (patient-facing)")
    summary_key = f"summary_{plan.id}"
    st.text_area(
        "Edit before approving",
        value=plan.week_summary,
        key=summary_key,
        height=80,
        label_visibility="collapsed",
    )

    # Exercise list / JSON editor
    st.subheader(f"Exercises ({len(plan.exercises)})")
    edit_mode_key = f"edit_mode_{plan.id}"
    if edit_mode_key not in st.session_state:
        st.session_state[edit_mode_key] = False

    json_key = f"json_{plan.id}"

    if st.session_state[edit_mode_key]:
        if json_key not in st.session_state:
            st.session_state[json_key] = json.dumps(plan.exercises, indent=2)
        st.text_area(
            "Edit exercises (JSON)",
            key=json_key,
            height=400,
        )
        if st.button("Validate JSON", key=f"preview_{plan.id}"):
            try:
                json.loads(st.session_state[json_key])
                st.success("JSON is valid.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
    else:
        # Store canonical JSON in session so approve can always read it
        st.session_state[json_key] = json.dumps(plan.exercises)
        for i, ex in enumerate(plan.exercises, start=1):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{i}. {ex['name']}**")
                    hold = f" — hold {ex['hold_seconds']}s" if ex.get("hold_seconds") else ""
                    st.write(f"{ex['sets']} sets × {ex['reps']}{hold}")
                    for cue in ex.get("cues", []):
                        st.write(f"- {cue}")
                with c2:
                    st.caption("RAG source")
                    st.code(ex.get("rag_source_id", "—"), language=None)
                if ex.get("rag_excerpt"):
                    with st.expander("Protocol excerpt"):
                        st.write(ex["rag_excerpt"])
                if ex.get("rationale"):
                    with st.expander("Rationale"):
                        st.write(ex["rationale"])
                if ex.get("contraindications"):
                    st.warning("Contraindications: " + "; ".join(ex["contraindications"]))

    st.toggle("Edit exercises (JSON)", key=edit_mode_key)

    # Approve / Reject
    st.divider()
    pt_notes = st.text_area(
        "PT notes (visible in audit log)",
        key=f"notes_{plan.id}",
        placeholder="Optional — add clinical context or reason for any changes.",
        height=60,
    )

    col_approve, col_reject = st.columns(2)

    with col_approve:
        if st.button("Approve plan", key=f"approve_{plan.id}", type="primary", use_container_width=True):
            try:
                edited_exercises = json.loads(st.session_state[json_key])
            except json.JSONDecodeError as e:
                st.error(f"Fix the exercise JSON before approving: {e}")
                return
            missing_citations = [
                ex["name"] for ex in edited_exercises
                if not ex.get("rag_source_id", "").strip()
            ]
            if missing_citations:
                st.error(
                    f"These exercises are missing RAG citations: {missing_citations}. "
                    "Add rag_source_id before approving."
                )
                return
            with get_db() as db:
                db.update_plan_exercises(
                    plan.id, edited_exercises, st.session_state[summary_key]
                )
                db.approve_plan(plan.id, pt_notes=pt_notes)
            from agent.state_machine import RehabStateMachine
            RehabStateMachine(plan.patient_id).trigger_plan_approved()
            st.success("Plan approved — patient will see it in their next session.")
            st.rerun()

    with col_reject:
        if st.button("Reject plan", key=f"reject_{plan.id}", use_container_width=True):
            if not pt_notes.strip():
                st.error("Enter a reason in the notes field before rejecting.")
            else:
                with get_db() as db:
                    db.reject_plan(plan.id, pt_notes=pt_notes)
                st.warning("Plan rejected. A new plan can be generated after addressing the issues.")
                st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_plans, tab_flags, tab_roster = st.tabs([
    f"Plan queue ({len(pending_plans)})",
    f"Red flags ({len(open_flags)})",
    f"Patients ({len(all_patients)})",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Plan approval queue
# ═══════════════════════════════════════════════════════════════════════════════
with tab_plans:
    if not pending_plans:
        st.success("No plans awaiting review.")
    else:
        st.caption(
            "Review each AI-generated plan before it reaches the patient. "
            "Every exercise shows its protocol source."
        )
        for plan, patient in pending_plans:
            header = (
                f"**{patient.name}** — {patient.protocol} — "
                f"Wk {plan.week_start}–{plan.week_end} — "
                f"generated {plan.generated_at.strftime('%Y-%m-%d %H:%M')}"
            )
            with st.expander(header, expanded=len(pending_plans) == 1):
                _render_plan_review(plan, patient)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Red-flag queue
# ═══════════════════════════════════════════════════════════════════════════════
with tab_flags:
    if not open_flags:
        st.success("No open red flags.")
    else:
        st.caption(f"{len(open_flags)} flag(s) awaiting review.")

    for flag in open_flags:
        label = (
            f"Patient {flag.patient_id} — "
            f"{flag.triggered_at.strftime('%Y-%m-%d %H:%M')} — "
            f"Pain {flag.pain_score}/10 — {flag.swelling} swelling"
        )
        with st.expander(label):
            st.write("**Flags triggered:**")
            for f_text in flag.flags:
                st.write(f"- {f_text}")
            st.write(f"Giving way: {'Yes' if flag.giving_way else 'No'}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Mark reviewed", key=f"rev_{flag.id}", use_container_width=True):
                    with get_db() as db:
                        db.mark_flag_reviewed(flag.id)
                    st.rerun()
            with col2:
                if st.button("Escalate", key=f"esc_{flag.id}", use_container_width=True):
                    with get_db() as db:
                        db.escalate_flag(flag.id)
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Patient roster
# ═══════════════════════════════════════════════════════════════════════════════
with tab_roster:
    st.subheader("Patient roster")
    if not all_patients:
        st.info("No patients registered yet.")

    for p in all_patients:
        with st.expander(f"{p.name} — {p.protocol} — Week {p.weeks_post_op}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Surgery: {p.surgery_date}   Side: {p.side}")
                st.write(f"Graft: {p.graft_type}   WB: {p.weight_bearing_status}")
                st.write(f"Meniscal repair: {p.meniscal_repair}")
            with col2:
                proto_options = [proto.value for proto in Protocol]
                new_protocol = st.selectbox(
                    "Override protocol",
                    options=proto_options,
                    index=proto_options.index(p.protocol),
                    key=f"proto_{p.id}",
                )
                if st.button("Apply override", key=f"apply_{p.id}"):
                    with get_db() as db:
                        db.update_protocol(p.id, new_protocol)
                    st.success("Protocol updated. Next plan generation will use the new protocol.")
                    st.rerun()
