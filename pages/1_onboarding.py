"""
Step 1 — Patient onboarding.

Required fields (all gated — no MISSING sentinels reach db.save_patient):
  graft_type, surgery_date, weight_bearing_status, meniscal_repair,
  stated_goal_text, side, protocol.

Validation runs before any DB write; each missing field gets a named error.
"""
from __future__ import annotations

from datetime import date

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
st.caption("Take a few minutes to set up your profile. Everything is stored on this device only.")

# ── Helper ────────────────────────────────────────────────────────────────────

def _enum_options(enum_cls) -> list:
    return [None] + list(enum_cls)


def _fmt(val) -> str:
    if val is None:
        return "— select —"
    if hasattr(val, "display"):
        return val.display()
    return str(val).replace("_", " ").title()


# ── Form ──────────────────────────────────────────────────────────────────────

with st.form("onboarding_form", border=True):
    st.subheader("About you")
    name = st.text_input("First name *")

    col_side, col_proto = st.columns(2)
    with col_side:
        side = st.selectbox(
            "Operated leg *",
            options=[None, "Left", "Right"],
            format_func=lambda x: "— select —" if x is None else x,
            index=0,
        )
    with col_proto:
        protocol = st.selectbox(
            "Assigned protocol *",
            options=_enum_options(Protocol),
            format_func=_fmt,
            index=0,
        )

    st.subheader("Surgery details")
    surgery_date = st.date_input(
        "Surgery date *",
        value=None,
        max_value=date.today(),
        help="Leave blank if not yet known — you can update this later.",
    )

    col_graft, col_wb = st.columns(2)
    with col_graft:
        graft_type = st.selectbox(
            "Graft type *",
            options=_enum_options(GraftType),
            format_func=_fmt,
            index=0,
        )
    with col_wb:
        weight_bearing_status = st.selectbox(
            "Current weight-bearing status *",
            options=_enum_options(WeightBearingStatus),
            format_func=_fmt,
            index=0,
        )

    meniscal_repair = st.selectbox(
        "Meniscal repair performed? *",
        options=list(MeniscalRepair),
        format_func=_fmt,
        index=0,           # defaults to NONE — always a valid answer
    )

    st.subheader("Your goal")
    stated_goal_text = st.text_area(
        "In your own words, what does getting back to full recovery mean for you? *",
        placeholder="e.g. I want to play in my club's pre-season in August, and be confident on the pitch again.",
        height=100,
        help="Minimum 5 characters. This is kept on your device only.",
    )

    pt_code = st.text_input(
        "PT access code (optional)",
        help="Your physio will give you this if they want to see your progress.",
    )

    submitted = st.form_submit_button("Save and continue", type="primary")

# ── Validation (runs after submit; nothing reaches DB until all checks pass) ──

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
    # meniscal_repair always has a value (NONE is a valid clinical answer)
    if not stated_goal_text.strip():
        errors.append("Please describe your recovery goal — even a single sentence helps.")
    elif len(stated_goal_text.strip()) < 5:
        errors.append("A little more detail in the goal field would be helpful.")

    if errors:
        for msg in errors:
            st.error(msg)
        st.stop()

    # ── Construct model (secondary validation via Pydantic) ───────────────────
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
        # Surface any Pydantic errors that slipped past the manual checks
        for err in exc.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            st.error(f"{field}: {err['msg']}")
        st.stop()

    # ── Persist ───────────────────────────────────────────────────────────────
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
