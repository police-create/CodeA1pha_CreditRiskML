"""Secondary Interactive Web Application using Streamlit.

Provides an alternative dashboard interface for credit-default risk assessment,
displaying model performance insights, threshold controls, and instant borrower scoring.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.demo_config import (
    EDUCATION_MAP,
    MARRIAGE_MAP,
    SEX_MAP,
    PAY_STATUS_MAP,
    PRESET_DEMO_SAMPLES,
)

MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
SCHEMA_PATH = PROJECT_ROOT / "outputs" / "results" / "input_schema.json"
METRICS_PATH = PROJECT_ROOT / "outputs" / "results" / "final_metrics.json"

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_model_and_schema():
    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        st.error("Model artifacts not found. Please run 'python src/train.py' first.")
        st.stop()
    pipeline = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return pipeline, schema


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


pipeline, schema = load_model_and_schema()
metrics_data = load_metrics()

st.title("💳 CREDIT SCORING MODEL")
st.markdown("### Supervised Machine Learning Classification for Credit-Default Risk")
st.info("⚠️ **Educational Machine Learning Prototype**: Developed for laboratory demonstration. Not a real lending authorization system.")

# Sidebar for Quick Demo Samples
st.sidebar.header("📋 Demo Samples")
selected_sample = st.sidebar.selectbox("Load Authentic Dataset Record:", list(PRESET_DEMO_SAMPLES.keys()))

if st.sidebar.button("Load Selected Sample"):
    st.session_state["loaded_sample"] = PRESET_DEMO_SAMPLES[selected_sample]
    st.sidebar.success("Sample loaded into form!")

# Default values resolution
current_vals = st.session_state.get("loaded_sample", PRESET_DEMO_SAMPLES[list(PRESET_DEMO_SAMPLES.keys())[0]])["values"]
current_gt = st.session_state.get("loaded_sample", PRESET_DEMO_SAMPLES[list(PRESET_DEMO_SAMPLES.keys())[0]])["actual_target"]

# Main tabs
tab_predict, tab_metrics, tab_about = st.tabs(["🚀 Live Prediction", "📊 Model Evaluation & Metrics", "📖 Methodology & Responsible ML"])

with tab_predict:
    st.subheader("Borrower Information Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Demographics & Line of Credit")
        limit_bal = st.number_input("Given Credit Limit (NTD)", min_value=1000, max_value=1000000, value=int(current_vals["LIMIT_BAL"]), step=5000)
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=int(current_vals["AGE"]))
        sex_options = list(SEX_MAP.keys())
        sex_idx = 0 if current_vals["SEX"] == 1 else 1
        sex_label = st.selectbox("Gender", sex_options, index=sex_idx)
        edu_options = list(EDUCATION_MAP.keys())
        edu_idx = min(current_vals["EDUCATION"] - 1, len(edu_options) - 1)
        edu_label = st.selectbox("Education Level", edu_options, index=edu_idx)
        marr_options = list(MARRIAGE_MAP.keys())
        marr_idx = min(current_vals["MARRIAGE"] - 1, len(marr_options) - 1)
        marriage_label = st.selectbox("Marital Status", marr_options, index=marr_idx)

    with col2:
        st.markdown("#### Repayment Status (Last 6 Months)")
        pay_options = list(PAY_STATUS_MAP.keys())

        def get_pay_idx(val):
            for i, (k, v) in enumerate(PAY_STATUS_MAP.items()):
                if v == val:
                    return i
            return 2  # default 0

        p0 = st.selectbox("Sept Status (PAY_0)", pay_options, index=get_pay_idx(current_vals["PAY_0"]))
        p2 = st.selectbox("Aug Status (PAY_2)", pay_options, index=get_pay_idx(current_vals["PAY_2"]))
        p3 = st.selectbox("Jul Status (PAY_3)", pay_options, index=get_pay_idx(current_vals["PAY_3"]))
        p4 = st.selectbox("Jun Status (PAY_4)", pay_options, index=get_pay_idx(current_vals["PAY_4"]))
        p5 = st.selectbox("May Status (PAY_5)", pay_options, index=get_pay_idx(current_vals["PAY_5"]))
        p6 = st.selectbox("Apr Status (PAY_6)", pay_options, index=get_pay_idx(current_vals["PAY_6"]))

    with col3:
        st.markdown("#### Recent Statements & Payments (NTD)")
        b1 = st.number_input("Sept Bill (BILL_AMT1)", value=float(current_vals["BILL_AMT1"]))
        b2 = st.number_input("Aug Bill (BILL_AMT2)", value=float(current_vals["BILL_AMT2"]))
        b3 = st.number_input("Jul Bill (BILL_AMT3)", value=float(current_vals["BILL_AMT3"]))
        pm1 = st.number_input("Sept Paid (PAY_AMT1)", min_value=0.0, value=float(current_vals["PAY_AMT1"]))
        pm2 = st.number_input("Aug Paid (PAY_AMT2)", min_value=0.0, value=float(current_vals["PAY_AMT2"]))
        pm3 = st.number_input("Jul Paid (PAY_AMT3)", min_value=0.0, value=float(current_vals["PAY_AMT3"]))

    # Additional billing fields in expander
    with st.expander("Show Remaining Billing / Payment Months (Apr - Jun)"):
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            b4 = st.number_input("Jun Bill (BILL_AMT4)", value=float(current_vals["BILL_AMT4"]))
            b5 = st.number_input("May Bill (BILL_AMT5)", value=float(current_vals["BILL_AMT5"]))
            b6 = st.number_input("Apr Bill (BILL_AMT6)", value=float(current_vals["BILL_AMT6"]))
        with c_exp2:
            pm4 = st.number_input("Jun Paid (PAY_AMT4)", min_value=0.0, value=float(current_vals["PAY_AMT4"]))
            pm5 = st.number_input("May Paid (PAY_AMT5)", min_value=0.0, value=float(current_vals["PAY_AMT5"]))
            pm6 = st.number_input("Apr Paid (PAY_AMT6)", min_value=0.0, value=float(current_vals["PAY_AMT6"]))

    if st.button("⚡ Assess Credit Default Risk", type="primary"):
        record = {
            "LIMIT_BAL": float(limit_bal),
            "SEX": int(SEX_MAP[sex_label]),
            "EDUCATION": int(EDUCATION_MAP[edu_label]),
            "MARRIAGE": int(MARRIAGE_MAP[marriage_label]),
            "AGE": int(age),
            "PAY_0": int(PAY_STATUS_MAP[p0]),
            "PAY_2": int(PAY_STATUS_MAP[p2]),
            "PAY_3": int(PAY_STATUS_MAP[p3]),
            "PAY_4": int(PAY_STATUS_MAP[p4]),
            "PAY_5": int(PAY_STATUS_MAP[p5]),
            "PAY_6": int(PAY_STATUS_MAP[p6]),
            "BILL_AMT1": float(b1),
            "BILL_AMT2": float(b2),
            "BILL_AMT3": float(b3),
            "BILL_AMT4": float(b4),
            "BILL_AMT5": float(b5),
            "BILL_AMT6": float(b6),
            "PAY_AMT1": float(pm1),
            "PAY_AMT2": float(pm2),
            "PAY_AMT3": float(pm3),
            "PAY_AMT4": float(pm4),
            "PAY_AMT5": float(pm5),
            "PAY_AMT6": float(pm6),
        }

        df_in = pd.DataFrame([record])[schema["feature_names"]]
        pred = int(pipeline.predict(df_in)[0])
        prob = float(pipeline.predict_proba(df_in)[0, 1])

        st.markdown("---")
        st.subheader("Assessment Results")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if pred == 1:
                st.error("🚨 **PREDICTION: DEFAULT RISK DETECTED**")
            else:
                st.success("✅ **PREDICTION: NO DEFAULT (LOW/MODERATE RISK)**")

        with res_col2:
            st.metric(label="Estimated Default Probability", value=f"{prob * 100:.2f}%")

        with res_col3:
            if prob < 0.25:
                tier = "🟢 Prime / Low Risk"
            elif prob < 0.50:
                tier = "🟡 Moderate Risk"
            elif prob < 0.70:
                tier = "🟠 Elevated Delinquency Risk"
            else:
                tier = "🔴 Severe Default Risk"
            st.write(f"**Risk Tier:** {tier}")

        st.caption(f"Historical Comparison: Actual target in dataset for this profile was: {'Default (1)' if current_gt == 1 else 'No Default (0)'}")

with tab_metrics:
    st.subheader("Model Performance Evaluation on Untouched Test Set (6,000 samples)")
    if metrics_data:
        df_models = pd.DataFrame(list(metrics_data["models"].values()))
        st.dataframe(df_models.style.highlight_max(subset=["Accuracy", "F1-Score", "ROC-AUC", "Average Precision"], color="#d4edda"), use_container_width=True)

        st.markdown("#### Operational Threshold Trade-off Analysis (Random Forest)")
        df_th = pd.DataFrame(metrics_data["threshold_analysis"])
        st.dataframe(df_th, use_container_width=True)

        st.markdown("#### Top Predictive Features (Gini Importance)")
        df_top = pd.DataFrame(metrics_data["top_features"])
        st.bar_chart(df_top.set_index("Feature"))
    else:
        st.warning("Final metrics JSON not found.")

with tab_about:
    st.subheader("Methodology & Responsible Machine Learning")
    st.markdown(
        """
        - **Dataset**: UCI Machine Learning Repository Dataset #350 (Default of Credit Card Clients).
        - **Target**: `default_payment_next_month` (Binary: 0 = No Default, 1 = Default).
        - **Leakage Prevention**: ID excluded from predictors; Preprocessing fitted strictly on training data; CV evaluated exclusively on training folds.
        - **Fairness & Demographics**: The dataset includes demographic attributes (`SEX`, `EDUCATION`, `MARRIAGE`). In production financial systems, using demographic attributes directly requires compliance with fair lending legislation (e.g., US Equal Credit Opportunity Act). In this educational prototype, they illustrate feature encoding and proxy analysis.
        - **Limitations**: Trained on Taiwanese credit card accounts from 2005; results do not generalize to other economies, time horizons, or loan structures.
        """
    )
