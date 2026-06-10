"""
Streamlit frontend for the Agentic Clinical Trial Recruitment system.

Talks to the FastAPI backend (app/api.py) over HTTP — clean separation of
the UI from the agent logic.

Run (with the API already running on port 8000):
    streamlit run app/dashboard.py

Start the backend first in another terminal:
    uvicorn app.api:app --reload --port 8000
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Clinical Trial Recruitment",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Agentic Clinical Trial Recruitment")
st.caption(
    "Screen patients against trial eligibility, predict dropout risk, "
    "generate compliant outreach — all through a LangGraph multi-agent backend."
)

# ---- Sidebar: backend health ----
with st.sidebar:
    st.header("Backend status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            st.success("✅ API connected")
        else:
            st.error("API reachable but not healthy")
    except Exception:
        st.error("API not reachable.\nStart it with:\n`uvicorn app.api:app --port 8000`")

# ---- Inputs ----
col1, col2 = st.columns(2)
with col1:
    patient_id = st.text_input("Patient ID", value="P001")
with col2:
    trial_id = st.text_input("Trial ID", value="NCT06710340")

# ---- Action ----
if st.button("Screen patient", type="primary"):
    with st.spinner("Running agentic pipeline (Screener → Communicator → Compliance)..."):
        try:
            resp = requests.post(
                f"{API_URL}/screen",
                json={"patient_id": patient_id, "trial_id": trial_id},
                timeout=120,
            )
            if resp.ok:
                data = resp.json()

                # Eligibility verdict
                if data.get("eligible") is True:
                    st.success("✅ Eligible")
                elif data.get("eligible") is False:
                    st.warning("❌ Not eligible")
                else:
                    st.info("Eligibility: not determined")

                # Scores
                m1, m2, m3 = st.columns(3)
                m1.metric("Eligibility score", data.get("eligibility_score", "—"))
                m2.metric("Dropout risk", data.get("dropout_risk", "—"))
                m3.metric("Final score", data.get("final_score", "—"))

                # Compliance
                st.subheader("Compliance gate")
                st.write(data.get("compliance_status", "N/A"))

                # Outreach message
                if data.get("outreach_message"):
                    st.subheader("Generated outreach")
                    st.info(data["outreach_message"])

                # Audit trail
                with st.expander("Full audit log"):
                    st.json(data.get("audit_log", {}))
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
