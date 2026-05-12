"""Step 4 — Progress view: pain trend, milestone tracking, weekly summary."""
import streamlit as st
from data.db import get_db

st.set_page_config(page_title="My Progress", page_icon="📈")
st.title("My Progress")

if "patient_id" not in st.session_state:
    st.warning("Please complete onboarding first.")
    st.switch_page("pages/1_onboarding.py")

patient_id = st.session_state.patient_id

with get_db() as db:
    sessions = db.get_sessions(patient_id)
    patient = db.get_patient(patient_id)

if not sessions:
    st.info("No sessions recorded yet. Complete your first session to see progress.")
    st.stop()

# ── Pain trend chart ─────────────────────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(
    [{"date": s.date, "pain": s.pain_score} for s in sessions]
).set_index("date")

st.subheader("Pain over time")
st.line_chart(df["pain"])

# ── Milestone checklist ──────────────────────────────────────────────────────
st.subheader("Protocol milestones")
with get_db() as db:
    milestones = db.get_milestones(patient_id)

for m in milestones:
    icon = "✅" if m.achieved else "⬜"
    st.write(f"{icon} {m.name} (week {m.target_week})")

# ── Weekly summary ───────────────────────────────────────────────────────────
st.subheader("This week")
week_sessions = [s for s in sessions if s.is_this_week()]
st.metric("Sessions completed", len(week_sessions))
if week_sessions:
    avg_pain = sum(s.pain_score for s in week_sessions) / len(week_sessions)
    st.metric("Avg pain score", f"{avg_pain:.1f} / 10")

if st.button("Start today's session"):
    st.switch_page("pages/2_daily_session.py")
