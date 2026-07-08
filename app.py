"""Streamlit front-end for the Telco churn model.

Two modes, switchable in the sidebar:
- "Local model": loads models/churn_model.joblib directly. Use this on
  Streamlit Community Cloud (commit the model file or train at startup).
- "Remote API": calls the deployed FastAPI endpoint, demonstrating the
  UI -> API -> model architecture. Paste the App Runner URL in the sidebar
  or set the CHURN_API_URL environment variable / Streamlit secret.

Run locally:
    streamlit run app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"
DEFAULT_API_URL = os.environ.get("CHURN_API_URL", "")

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")

st.title("Telco Customer Churn Predictor")
st.caption(
    "Class-weighted logistic regression · ROC-AUC 0.842 · trained on the IBM Telco dataset. "
    "Adjust the customer profile and see churn risk update."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Prediction backend")
    mode = st.radio(
        "Mode",
        ["Local model", "Remote API"],
        help="Local loads the joblib pipeline in-process. Remote calls the deployed FastAPI service.",
    )
    api_url = ""
    if mode == "Remote API":
        api_url = st.text_input(
            "API base URL",
            value=DEFAULT_API_URL,
            placeholder="https://xxxx.awsapprunner.com",
        ).rstrip("/")

    st.divider()
    st.markdown(
        "**Project links**\n\n"
        "[Source on GitHub](https://github.com/A-jcodes) · "
        "[Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

# ---------------------------------------------------------------- inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    monthly = st.number_input("Monthly charges ($)", 15.0, 150.0, 85.0, step=5.0)
    total = st.number_input(
        "Total charges ($)",
        0.0,
        10000.0,
        float(round(monthly * max(tenure, 1), 2)),
        step=50.0,
        help="Defaults to monthly x tenure; override if known.",
    )

with col2:
    st.subheader("Customer & services")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = st.selectbox("Phone service", ["Yes", "No"])

    no_internet = internet == "No"
    svc_options = ["No", "Yes"] if not no_internet else ["No internet service"]

    def svc(label: str) -> str:
        if no_internet:
            return "No internet service"
        return st.selectbox(label, ["No", "Yes"], key=label)

    tech_support = svc("Tech support")
    online_security = svc("Online security")
    online_backup = svc("Online backup")
    device_protection = svc("Device protection")
    streaming_tv = svc("Streaming TV")
    streaming_movies = svc("Streaming movies")

multiple_lines = "No phone service" if phone == "No" else st.selectbox(
    "Multiple lines", ["No", "Yes"]
)

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
}

# ---------------------------------------------------------------- predict
def predict_local(record: dict) -> dict:
    import joblib

    @st.cache_resource
    def _load():
        return joblib.load(MODEL_PATH)

    proba = float(_load().predict_proba(pd.DataFrame([record]))[0, 1])
    band = "high" if proba >= 0.6 else "medium" if proba >= 0.3 else "low"
    return {"churn_probability": proba, "churn_predicted": proba >= 0.5, "risk_band": band}


def predict_remote(record: dict, base_url: str) -> dict:
    import requests

    resp = requests.post(f"{base_url}/predict", json=record, timeout=15)
    resp.raise_for_status()
    return resp.json()


st.divider()
if st.button("Predict churn risk", type="primary", use_container_width=True):
    try:
        if mode == "Remote API":
            if not api_url:
                st.error("Enter the deployed API URL in the sidebar first.")
                st.stop()
            result = predict_remote(customer, api_url)
            source = f"live API at `{api_url}`"
        else:
            if not MODEL_PATH.exists():
                st.error("Model not found. Run `python -m src.train` first.")
                st.stop()
            result = predict_local(customer)
            source = "local model"
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    proba = result["churn_probability"]
    band = result["risk_band"]
    color = {"low": "green", "medium": "orange", "high": "red"}[band]

    m1, m2 = st.columns(2)
    m1.metric("Churn probability", f"{proba:.1%}")
    m2.markdown(f"### Risk band: :{color}[{band.upper()}]")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(f"Scored by the {source}.")

    if band == "high":
        st.warning(
            "Retention playbook: prioritize outreach. Typical levers for this profile: "
            "contract upgrade incentive, autopay discount, or a tech support add-on."
        )
    elif band == "medium":
        st.info("Monitor. Consider a light-touch offer at next billing cycle.")
    else:
        st.success("Low risk. No intervention needed.")
