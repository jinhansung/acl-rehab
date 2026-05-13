"""
Step 3 — Daily session.

Flow:
  1. FSM gate — redirect if not ACTIVE
  2. Check-in (NRS pain, swelling, giving-way, optional flags)
  3. Red-flag evaluation (deterministic, no LLM)
     → RED_FLAG state + hard stop if any flag raised
  4. Exercise checklist (from PT-approved plan)
  5. Post-session RPE + shared notes + private encrypted journal
  6. Save SessionRecord + Measurements
"""
from __future__ import annotations

from datetime import date

import streamlit as st
from cryptography.fernet import InvalidToken

from agent.red_flags import check_red_flags
from agent.state_machine import RehabState, RehabStateMachine
from data.db import get_db
from data.journal import save_journal_entry, verify_passphrase
from data.models import (
    Measurement,
    RedFlagEvent,
    SessionRecord,
    SwellingLevel,
)

st.set_page_config(page_title="Today's Session", page_icon="🏃")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("check_in_done", False),
    ("red_flags_raised", []),
    ("exercise_states", {}),   # {exercise_name: {completed, pain_during, notes}}
    ("session_saved", False),
    ("journal_passphrase", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Route guard ───────────────────────────────────────────────────────────────
if "patient_id" not in st.session_state:
    st.warning("Please complete onboarding first.")
    st.switch_page("pages/1_onboarding.py")

patient_id: int = st.session_state.patient_id
fsm = RehabStateMachine(patient_id)
state = fsm.get_state()

with get_db() as db:
    patient = db.get_patient(patient_id)
    plan = db.get_latest_plan(patient_id)

if state == RehabState.ONBOARDING:
    st.subheader("Generate your first plan")
    st.write(
        "Your profile is saved. To create your personalised exercise plan, "
        "we need to send some anonymised clinical details to the AI model."
    )

    with st.expander("What gets sent to the AI?"):
        st.write(
            "- Operated leg, graft type, weeks post-op, weight-bearing status, "
            "meniscal repair, assigned protocol  \n"
            "- Your stated recovery goal (your own words)  \n"
            "**What is never sent:** your name, date of birth, or any other identifying information."
        )

    if "plan_consent_given" not in st.session_state:
        st.session_state.plan_consent_given = False

    if not st.session_state.plan_consent_given:
        if st.button("I understand — generate my plan", type="primary"):
            st.session_state.plan_consent_given = True
            st.rerun()
        st.stop()

    # Consent given — save ConsentRecord then call API
    import hashlib
    from agent.tools import generate_plan
    from data.models import ConsentRecord, ConsentType

    anon_payload = (
        f"{patient.side}|{patient.graft_type}|{patient.weeks_post_op}|"
        f"{patient.weight_bearing_status}|{patient.meniscal_repair}|"
        f"{patient.protocol}|{patient.stated_goal_text}"
    )
    payload_hash = hashlib.sha256(anon_payload.encode()).hexdigest()

    consent = ConsentRecord(
        patient_id=patient_id,
        consent_type=ConsentType.PLAN_GENERATION,
        model_used="claude-sonnet-4-20250514",
        data_sent_hash=payload_hash,
    )

    with st.spinner("Generating your plan — this takes about 20 seconds…"):
        try:
            with get_db() as db:
                consent_id = db.save_consent(consent)
            week = patient.weeks_post_op
            plan_obj = generate_plan(
                patient=patient,
                consent_record_id=consent_id,
                week_start=week,
                week_end=week + 1,
            )
            with get_db() as db:
                db.save_rehab_plan(plan_obj)
            st.success(
                "Plan generated and sent to your PT for review. "
                "Come back once they've approved it."
            )
            st.session_state.plan_consent_given = False
        except Exception as exc:
            st.error(f"Plan generation failed: {exc}")
    st.stop()

if state == RehabState.PLAN_PENDING_PT:
    st.info(
        "Your plan has been sent to your PT for review. "
        "Come back once they've approved it — usually within 24 hours."
    )
    st.stop()

if state == RehabState.RED_FLAG:
    st.error(
        "A concern from your last session is being reviewed by your PT. "
        "Please contact them before continuing any exercises."
    )
    if st.session_state.red_flags_raised:
        for flag in st.session_state.red_flags_raised:
            st.write(f"- {flag}")
    st.stop()

if state == RehabState.COMPLETED:
    st.success("You have completed your rehabilitation protocol. Congratulations on your hard work.")
    st.stop()

# ── ACTIVE state ──────────────────────────────────────────────────────────────
if plan is None or plan.review_status != "approved":
    st.warning("No approved plan found. Your PT may still be reviewing it.")
    st.stop()

st.title("Today's Session")
st.caption(
    f"Week {patient.weeks_post_op} · {patient.protocol} · "
    f"{date.today().strftime('%A, %d %b %Y')}"
)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Check-in
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.check_in_done:
    st.subheader("How are you feeling today?")

    pain = st.slider(
        "Pain right now (NRS 0–10)",
        min_value=0, max_value=10, value=0,
        help="0 = no pain at all   10 = the worst pain you can imagine",
    )

    swelling = st.selectbox(
        "Swelling around the knee",
        options=[s.value for s in SwellingLevel],
        index=0,
    )

    giving_way = st.checkbox("Any giving-way or instability since your last session?")

    with st.expander("Other symptoms (tap to expand)"):
        fever = st.checkbox("Fever or chills")
        wound_drainage = st.checkbox("Wound discharge or unusual redness")

    if st.button("Submit check-in", type="primary"):
        flags = check_red_flags(
            pain=pain,
            swelling=swelling,
            giving_way=giving_way,
            fever=fever if "fever" in dir() else False,
            wound_drainage=wound_drainage if "wound_drainage" in dir() else False,
        )

        if flags:
            # Persist red-flag event and transition FSM — deterministic, no LLM
            flag_event = RedFlagEvent(
                patient_id=patient_id,
                flags=[str(f) for f in flags],
                pain_score=pain,
                swelling=SwellingLevel(swelling),
                giving_way=giving_way,
            )
            with get_db() as db:
                db.save_red_flag(flag_event)
            fsm.trigger_red_flag()
            st.session_state.red_flags_raised = flags
            st.rerun()

        # Save check-in measurements for trending
        with get_db() as db:
            db.save_measurement(Measurement(
                patient_id=patient_id,
                metric="pain_nrs",
                value=float(pain),
                unit="0-10",
            ))

        st.session_state.check_in_pain = pain
        st.session_state.check_in_swelling = swelling
        st.session_state.check_in_giving_way = giving_way
        st.session_state.check_in_done = True
        st.rerun()

    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Exercise checklist
# ═════════════════════════════════════════════════════════════════════════════
if not st.session_state.session_saved:
    st.subheader(f"Exercises — {len(plan.exercises)} today")

    if plan.week_summary:
        st.info(plan.week_summary)

    for ex in plan.exercises:
        name = ex["name"]
        if name not in st.session_state.exercise_states:
            st.session_state.exercise_states[name] = {
                "completed": False,
                "pain_during": 0,
                "notes": "",
            }

        with st.container(border=True):
            cols = st.columns([0.05, 0.95])
            with cols[0]:
                done = st.checkbox(
                    "Done", value=st.session_state.exercise_states[name]["completed"],
                    key=f"done_{name}", label_visibility="collapsed",
                )
                st.session_state.exercise_states[name]["completed"] = done
            with cols[1]:
                hold = f" · hold {ex['hold_seconds']}s" if ex.get("hold_seconds") else ""
                st.markdown(f"**{name}** — {ex['sets']} × {ex['reps']}{hold}")
                for cue in ex.get("cues", []):
                    st.caption(f"• {cue}")

                with st.expander("Pain during / notes"):
                    pain_ex = st.slider(
                        "Pain during this exercise (0–10)",
                        0, 10,
                        value=st.session_state.exercise_states[name]["pain_during"],
                        key=f"pain_{name}",
                    )
                    st.session_state.exercise_states[name]["pain_during"] = pain_ex

                    ex_note = st.text_input(
                        "Notes for this exercise",
                        value=st.session_state.exercise_states[name]["notes"],
                        key=f"note_{name}",
                        placeholder="How did it feel?",
                    )
                    st.session_state.exercise_states[name]["notes"] = ex_note

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Post-session wrap-up
    # ═════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Wrapping up")

    # RPE (Borg CR10)
    RPE_LABELS = {
        0: "Nothing at all", 1: "Very light", 2: "Light", 3: "Moderate",
        4: "Somewhat hard", 5: "Hard", 6: "Hard+", 7: "Very hard",
        8: "Very hard+", 9: "Very, very hard", 10: "Maximum effort",
    }
    rpe = st.select_slider(
        "How hard was today's session overall? (RPE)",
        options=list(RPE_LABELS.keys()),
        format_func=lambda v: f"{v} — {RPE_LABELS[v]}",
        value=5,
    )

    # Shared notes (PT-visible)
    st.write("**Session notes**")
    share_with_pt = st.toggle("Share these notes with my PT", value=True)
    shared_notes = st.text_area(
        "Notes" if share_with_pt else "Notes (private — PT will not see these)",
        placeholder="Anything to flag from today's session?",
        height=100,
        label_visibility="collapsed",
    )

    # Private encrypted journal
    st.divider()
    st.write("**Private journal** — encrypted on this device, never shared")

    # Passphrase setup / unlock
    if st.session_state.journal_passphrase is None:
        with st.form("passphrase_form"):
            st.caption(
                "Your journal is encrypted with a passphrase only you know. "
                "Enter the same passphrase each session to keep entries readable."
            )
            passphrase_input = st.text_input(
                "Journal passphrase", type="password",
                placeholder="At least 8 characters",
            )
            submitted = st.form_submit_button("Unlock journal")

        if submitted:
            if len(passphrase_input) < 8:
                st.error("Passphrase must be at least 8 characters.")
            else:
                # Verify against existing entries if any
                if not verify_passphrase(patient_id, passphrase_input):
                    st.error(
                        "That passphrase does not match your previous journal entries. "
                        "Try again, or skip the journal for today."
                    )
                else:
                    st.session_state.journal_passphrase = passphrase_input
                    st.rerun()
        st.caption("_Skip the journal for now — finish session without it._")
    else:
        journal_text = st.text_area(
            "Journal entry",
            placeholder="How are you feeling about your recovery today?",
            height=120,
            label_visibility="collapsed",
        )
        st.caption("Encrypted before saving. Your PT cannot read this.")

    # Finish session button
    st.divider()
    if st.button("Finish and save session", type="primary"):
        ex_states = st.session_state.exercise_states
        completed = [n for n, s in ex_states.items() if s["completed"]]
        skipped   = [n for n, s in ex_states.items() if not s["completed"]]

        record = SessionRecord(
            patient_id=patient_id,
            week_number=patient.weeks_post_op,
            pain_score=st.session_state.check_in_pain,
            swelling=SwellingLevel(st.session_state.check_in_swelling),
            giving_way=st.session_state.check_in_giving_way,
            exercises_completed=completed,
            exercises_skipped=skipped,
            session_notes=shared_notes if share_with_pt else "",
        )

        with get_db() as db:
            session_id = db.save_session(record)
            db.save_measurement(Measurement(
                patient_id=patient_id,
                session_id=session_id,
                metric="rpe_cr10",
                value=float(rpe),
                unit="0-10",
            ))
            # Per-exercise pain measurements
            for ex_name, ex_state in ex_states.items():
                if ex_state["completed"]:
                    db.save_measurement(Measurement(
                        patient_id=patient_id,
                        session_id=session_id,
                        metric=f"pain_during_{ex_name.lower().replace(' ', '_')}",
                        value=float(ex_state["pain_during"]),
                        unit="0-10",
                    ))

        # Save journal entry (encrypted, never shared) if entered
        passphrase = st.session_state.journal_passphrase
        journal_entry = locals().get("journal_text", "").strip()
        if passphrase and journal_entry:
            save_journal_entry(patient_id, journal_entry, passphrase)

        st.session_state.session_saved = True
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Post-save confirmation
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.session_saved:
    ex_states = st.session_state.exercise_states
    completed_count = sum(1 for s in ex_states.values() if s["completed"])
    total_count     = len(ex_states)

    st.success(f"Session saved — {completed_count}/{total_count} exercises completed.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("View my progress", use_container_width=True):
            st.switch_page("pages/3_progress.py")
    with col2:
        if st.button("Back to home", use_container_width=True):
            # Reset for next session
            for key in ["check_in_done", "exercise_states", "session_saved",
                        "red_flags_raised", "journal_passphrase"]:
                st.session_state.pop(key, None)
            st.switch_page("app.py")
