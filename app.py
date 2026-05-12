"""Entry point — routes to patient view or PT dashboard based on role."""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="ACL Rehab Coach",
    page_icon="🏃",
    layout="centered",
)

if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.title("ACL Rehab Coach")
    st.write("Who are you?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("I'm a Patient", use_container_width=True):
            st.session_state.role = "patient"
            st.switch_page("pages/1_onboarding.py")
    with col2:
        if st.button("I'm a Physical Therapist", use_container_width=True):
            st.session_state.role = "pt"
            st.switch_page("pages/4_pt_dashboard.py")
elif st.session_state.role == "patient":
    st.switch_page("pages/2_daily_session.py")
else:
    st.switch_page("pages/4_pt_dashboard.py")
