"""
Step 1 — Patient onboarding.

Features:
- If already onboarded, shows submitted data + continue / reset options.
- "Fill test data" button pre-populates every field for quick testing.
- Full form with Pydantic validation before any DB write.
"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st
from pydantic import ValidationError

from data.db import get_db
from data.models import (
    GraftType,
    MeniscalRepair,
    PatientProfile,
    Protocol,
    WeightBearingStatus,
)

st.set_page_config(page_title="Getting started", page_icon="📋")
st.title("Getting started")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _enum_options(enum_cls) -> list:
    return [None] + list(enum_cls)


def _fmt(val) -> str:
    if val is None:
        return "— select —"
    if hasattr(val, "display"):
        return val.display()
    return str(val).replace("_", " ").title()


# ── Already onboarded view ────────────────────────────────────────────────────

if "patient_id" in st.session_state:
    with get_db() as db:
        patient = db.get_patient(st.session_state.patient_id)

    if patient:
        st.success(f"Onboarded as **{patient.name}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Protocol", patient.protocol)
        c2.metric("Week post-op", patient.weeks_post_op)
        c3.metric("Graft", patient.graft_type.replace("_", " ").title() if isinstance(patient.graft_type, str) else patient.graft_type.display())
        c4.metric("Side", patient.side)

        st.table({
            "Surgery date":       str(patient.surgery_date),
            "Weight bearing":     patient.weight_bearing_status.replace("_", " ").title() if isinstance(patient.weight_bearing_status, str) else patient.weight_bearing_status.display(),
            "Meniscal repair":    patient.meniscal_repair.title() if isinstance(patient.meniscal_repair, str) else patient.meniscal_repair.display(),
            "Stated goal":        patient.stated_goal_text,
            "PT code":            patient.pt_code or "—",
        })

        st.divider()
        col_go, col_reset = st.columns(2)
        with col_go:
            if st.button("Continue to session", type="primary", use_container_width=True):
                st.switch_page("pages/2_daily_session.py")
        with col_reset:
            if st.button("Reset onboarding", type="secondary", use_container_width=True):
                for key in ["patient_id", "role", "protocol", "plan_consent_given",
                            "check_in_done", "exercise_states", "session_saved",
                            "red_flags_raised", "journal_passphrase"]:
                    st.session_state.pop(key, None)
                st.rerun()
        st.stop()


# ── Test-data prefill (outside form so it can trigger rerun) ─────────────────

st.caption("Take a few minutes to set up your profile. Everything is stored on this device only.")

if st.button("Fill test data", help="Pre-populates every field — useful during development"):
    st.session_state["onb_name"]          = "Test Patient"
    st.session_state["onb_side"]          = "Left"
    st.session_state["onb_protocol"]      = Protocol.MOON
    st.session_state["onb_surgery_date"]  = date.today() - timedelta(weeks=8)
    st.session_state["onb_graft"]         = GraftType.HAMSTRING
    st.session_state["onb_wb"]            = WeightBearingStatus.FULL
    st.session_state["onb_meniscal"]      = MeniscalRepair.NONE
    st.session_state["onb_goal"]          = (
        "Return to recreational football and feel confident cutting and sprinting again."
    )
    st.session_state["onb_pt_code"]       = ""
    st.rerun()

st.divider()

# ── Form ──────────────────────────────────────────────────────────────────────

with st.form("onboarding_form", border=True):
    st.subheader("About you")
    name = st.text_input("First name *", key="onb_name")

    col_side, col_proto = st.columns(2)
    with col_side:
        side = st.selectbox(
            "Operated leg *",
            options=[None, "Left", "Right"],
            format_func=lambda x: "— select —" if x is None else x,
            key="onb_side",
        )
    with col_proto:
        protocol = st.selectbox(
            "Assigned protocol *",
            options=_enum_options(Protocol),
            format_func=_fmt,
            key="onb_protocol",
        )

    st.subheader("Surgery details")
    surgery_date = st.date_input(
        "Surgery date *",
        value=None,
        max_value=date.today(),
        help="Leave blank if not yet known — you can update this later.",
        key="onb_surgery_date",
    )

    col_graft, col_wb = st.columns(2)
    with col_graft:
        graft_type = st.selectbox(
            "Graft type *",
            options=_enum_options(GraftType),
            format_func=_fmt,
            key="onb_graft",
        )
    with col_wb:
        weight_bearing_status = st.selectbox(
            "Current weight-bearing status *",
            options=_enum_options(WeightBearingStatus),
            format_func=_fmt,
            key="onb_wb",
        )

    meniscal_repair = st.selectbox(
        "Meniscal repair performed? *",
        options=list(MeniscalRepair),
        format_func=_fmt,
        key="onb_meniscal",
    )

    st.subheader("Your goal")
    stated_goal_text = st.text_area(
        "In your own words, what does getting back to full recovery mean for you? *",
        placeholder="e.g. I want to play in my club's pre-season in August.",
        height=100,
        help="Minimum 5 characters. Stored on your device only.",
        key="onb_goal",
    )

    pt_code = st.text_input(
        "PT access code (optional)",
        help="Your physio will give you this if they want to see your progress.",
        key="onb_pt_code",
    )

    submitted = st.form_submit_button("Save and continue", type="primary")

# ── Validation ────────────────────────────────────────────────────────────────

if submitted:
    errors: list[str] = []

    if not name.strip():
        errors.append("First name is required.")
    if side is None:
        errors.append("Operated leg is required.")
    if protocol is None:
        errors.append("Assigned protocol is required.")
    if surgery_date is None:
        errors.append("Surgery date is required.")
    if graft_type is None:
        errors.append("Graft type is required.")
    if weight_bearing_status is None:
        errors.append("Weight-bearing status is required.")
    if not stated_goal_text.strip():
        errors.append("Please describe your recovery goal — even a single sentence helps.")
    elif len(stated_goal_text.strip()) < 5:
        errors.append("A little more detail in the goal field would be helpful.")

    if errors:
        for msg in errors:
            st.error(msg)
        st.stop()

    try:
        profile = PatientProfile(
            name=name.strip(),
            side=side,
            graft_type=graft_type,
            surgery_date=surgery_date,
            weight_bearing_status=weight_bearing_status,
            meniscal_repair=meniscal_repair,
            stated_goal_text=stated_goal_text.strip(),
            protocol=protocol,
            pt_code=pt_code.strip() or None,
        )
    except ValidationError as exc:
        for err in exc.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            st.error(f"{field}: {err['msg']}")
        st.stop()

    try:
        with get_db() as db:
            patient_id = db.save_patient(profile)
    except Exception as exc:
        st.error(f"Could not save your profile: {exc}")
        st.stop()

    st.session_state.patient_id = patient_id
    st.session_state.role = "patient"
    st.session_state.protocol = profile.protocol
    st.success("Profile saved. Let's get started.")
    st.switch_page("pages/2_daily_session.py")
