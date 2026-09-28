"""Primary Classroom Demonstration Application using Gradio.

Title: CREDIT SCORING MODEL
Loads the serialized end-to-end pipeline (models/final_model.joblib), performs schema
and value range validation, supports one-click genuine demo profile evaluation,
and displays instant probability risk assessments with clear educational disclaimers.
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Ensure project root in sys.path
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


def load_model_and_schema():
    """Load serialized pipeline and validation schema."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}. Train the model first.")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file missing: {SCHEMA_PATH}.")

    pipeline = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    return pipeline, schema


# Global cache
MODEL_PIPELINE, INPUT_SCHEMA = load_model_and_schema()


def predict_credit_risk(
    limit_bal, age, sex_label, edu_label, marriage_label,
    pay_0_label, pay_2_label, pay_3_label, pay_4_label, pay_5_label, pay_6_label,
    bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
    pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
    demo_ground_truth=None,
):
    """Callback function for Gradio risk assessment."""
    # 1. Validation checks
    errors = []
    if limit_bal is None or limit_bal <= 0:
        errors.append("• Credit Limit (LIMIT_BAL) must be a positive number greater than 0 NTD.")
    if age is None or age < 18 or age > 100:
        errors.append("• Age must be an adult between 18 and 100 years.")
    if any(p is not None and p < 0 for p in [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]):
        errors.append("• Payment amounts cannot be negative.")

    if errors:
        error_report = "### ⚠️ Input Validation Notice\n" + "\n".join(errors)
        return (
            "VALIDATION FAILED",
            "0.00%",
            error_report,
        )

    # 2. Map Categorical / Ordinal Dropdown Labels to Numeric Codes
    sex_val = SEX_MAP.get(sex_label, 2)
    edu_val = EDUCATION_MAP.get(edu_label, 2)
    marr_val = MARRIAGE_MAP.get(marriage_label, 2)

    pay_0_val = PAY_STATUS_MAP.get(pay_0_label, 0)
    pay_2_val = PAY_STATUS_MAP.get(pay_2_label, 0)
    pay_3_val = PAY_STATUS_MAP.get(pay_3_label, 0)
    pay_4_val = PAY_STATUS_MAP.get(pay_4_label, 0)
    pay_5_val = PAY_STATUS_MAP.get(pay_5_label, 0)
    pay_6_val = PAY_STATUS_MAP.get(pay_6_label, 0)

    # 3. Assemble Record DataFrame strictly conforming to input schema
    record = {
        "LIMIT_BAL": float(limit_bal),
        "SEX": int(sex_val),
        "EDUCATION": int(edu_val),
        "MARRIAGE": int(marr_val),
        "AGE": int(age),
        "PAY_0": int(pay_0_val),
        "PAY_2": int(pay_2_val),
        "PAY_3": int(pay_3_val),
        "PAY_4": int(pay_4_val),
        "PAY_5": int(pay_5_val),
        "PAY_6": int(pay_6_val),
        "BILL_AMT1": float(bill_amt1 or 0.0),
        "BILL_AMT2": float(bill_amt2 or 0.0),
        "BILL_AMT3": float(bill_amt3 or 0.0),
        "BILL_AMT4": float(bill_amt4 or 0.0),
        "BILL_AMT5": float(bill_amt5 or 0.0),
        "BILL_AMT6": float(bill_amt6 or 0.0),
        "PAY_AMT1": float(pay_amt1 or 0.0),
        "PAY_AMT2": float(pay_amt2 or 0.0),
        "PAY_AMT3": float(pay_amt3 or 0.0),
        "PAY_AMT4": float(pay_amt4 or 0.0),
        "PAY_AMT5": float(pay_amt5 or 0.0),
        "PAY_AMT6": float(pay_amt6 or 0.0),
    }

    df_in = pd.DataFrame([record])[INPUT_SCHEMA["feature_names"]]

    # 4. Predict using the serialized pipeline
    pred_class = int(MODEL_PIPELINE.predict(df_in)[0])
    prob_default = float(MODEL_PIPELINE.predict_proba(df_in)[0, 1])

    risk_label = "DEFAULT RISK DETECTED" if pred_class == 1 else "NO DEFAULT (LOW/MODERATE RISK)"
    prob_pct = f"{prob_default * 100:.2f}%"

    # Detail summary
    if prob_default < 0.25:
        tier = "Tier 1: Prime / Low Risk"
        badge = "🟢"
    elif prob_default < 0.50:
        tier = "Tier 2: Moderate Risk"
        badge = "🟡"
    elif prob_default < 0.70:
        tier = "Tier 3: Elevated Risk"
        badge = "🟠"
    else:
        tier = "Tier 4: Severe Default Risk"
        badge = "🔴"

    details = (
        f"### {badge} Assessment Summary\n"
        f"- **Risk Classification**: {risk_label}\n"
        f"- **Estimated Default Probability**: **{prob_pct}**\n"
        f"- **Assigned Risk Tier**: {tier}\n"
        f"- **Decision Cut-off**: 50.00% (Standard Operational Threshold)\n"
    )

    if demo_ground_truth is not None and str(demo_ground_truth).strip() != "":
        actual_name = "Default (1)" if int(demo_ground_truth) == 1 else "No Default (0)"
        details += f"\n---\n**Educational Comparison**:\n- **Actual Historical Dataset Label**: `{actual_name}`\n"
        if (int(demo_ground_truth) == 1 and pred_class == 1) or (int(demo_ground_truth) == 0 and pred_class == 0):
            details += "- **Outcome**: Model prediction matches historical record! ✅\n"
        else:
            details += "- **Outcome**: Model prediction differs from actual label (demonstrating real-world classification trade-offs). ⚖️\n"

    details += (
        "\n> **Educational Disclaimer**:\n"
        "> This is an educational machine-learning prototype developed for college laboratory demonstration. "
        "It is NOT a real lending or financial credit authorization system."
    )

    return risk_label, prob_pct, details


def populate_demo_profile(sample_key):
    """Populate form inputs with authentic dataset profile."""
    if sample_key not in PRESET_DEMO_SAMPLES:
        return [gr.update()] * 24

    sample = PRESET_DEMO_SAMPLES[sample_key]
    vals = sample["values"]
    gt = sample["actual_target"]

    # Invert maps to find dropdown labels
    inv_sex = {v: k for k, v in SEX_MAP.items()}
    inv_edu = {v: k for k, v in EDUCATION_MAP.items()}
    inv_marr = {v: k for k, v in MARRIAGE_MAP.items()}
    inv_pay = {v: k for k, v in PAY_STATUS_MAP.items()}

    return [
        vals["LIMIT_BAL"],
        vals["AGE"],
        inv_sex[vals["SEX"]],
        inv_edu[vals["EDUCATION"]],
        inv_marr[vals["MARRIAGE"]],
        inv_pay[vals["PAY_0"]],
        inv_pay[vals["PAY_2"]],
        inv_pay[vals["PAY_3"]],
        inv_pay[vals["PAY_4"]],
        inv_pay[vals["PAY_5"]],
        inv_pay[vals["PAY_6"]],
        vals["BILL_AMT1"],
        vals["BILL_AMT2"],
        vals["BILL_AMT3"],
        vals["BILL_AMT4"],
        vals["BILL_AMT5"],
        vals["BILL_AMT6"],
        vals["PAY_AMT1"],
        vals["PAY_AMT2"],
        vals["PAY_AMT3"],
        vals["PAY_AMT4"],
        vals["PAY_AMT5"],
        vals["PAY_AMT6"],
        gt,  # hidden demo ground truth state
    ]


# Build Gradio UI
with gr.Blocks(title="CREDIT SCORING MODEL") as demo:
    demo_truth_state = gr.State(value=None)

    gr.Markdown(
        """
        # 🏦 CREDIT SCORING MODEL
        ### Supervised Machine Learning Classification for Credit-Default Risk
        *College Laboratory Prototype | Powered by Trained Pipeline (`models/final_model.joblib`)*
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Quick Demo Selector")
            sample_dropdown = gr.Dropdown(
                choices=list(PRESET_DEMO_SAMPLES.keys()),
                label="Choose an Authentic Dataset Record for Classroom Demo",
                value=list(PRESET_DEMO_SAMPLES.keys())[0],
            )
            load_demo_btn = gr.Button("📂 Load Selected Demo Sample", variant="secondary")

        with gr.Column(scale=3):
            gr.Markdown(
                """
                > **Notice**: Predictions are generated by the complete serialized pipeline fitted on 24,000 real borrower records from the UCI Credit Card Default repository.
                > No hardcoded results.
                """
            )

    with gr.Tabs():
        with gr.TabItem("1. Borrower Profile & Status"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Demographics & Line of Credit")
                    in_limit = gr.Number(label="Given Credit Limit (NT Dollar)", value=260000, minimum=1000)
                    in_age = gr.Number(label="Age (Years)", value=32, minimum=18, maximum=100)
                    in_sex = gr.Dropdown(choices=list(SEX_MAP.keys()), label="Gender", value="Female (2)")
                    in_edu = gr.Dropdown(choices=list(EDUCATION_MAP.keys()), label="Education Level", value="Graduate School (1)")
                    in_marriage = gr.Dropdown(choices=list(MARRIAGE_MAP.keys()), label="Marital Status", value="Single (2)")

                with gr.Column():
                    gr.Markdown("#### Recent Payment Status (Delinquency History)")
                    in_pay0 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="September Repayment Status (PAY_0)", value="-1: Paid in full")
                    in_pay2 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="August Repayment Status (PAY_2)", value="-1: Paid in full")
                    in_pay3 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="July Repayment Status (PAY_3)", value="-1: Paid in full")
                    in_pay4 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="June Repayment Status (PAY_4)", value="-1: Paid in full")
                    in_pay5 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="May Repayment Status (PAY_5)", value="-1: Paid in full")
                    in_pay6 = gr.Dropdown(choices=list(PAY_STATUS_MAP.keys()), label="April Repayment Status (PAY_6)", value="-1: Paid in full")

        with gr.TabItem("2. Billing Statement History"):
            gr.Markdown("#### Monthly Statement Amounts (NT Dollar)")
            with gr.Row():
                in_b1 = gr.Number(label="Sept Bill (BILL_AMT1)", value=2500)
                in_b2 = gr.Number(label="Aug Bill (BILL_AMT2)", value=2600)
                in_b3 = gr.Number(label="Jul Bill (BILL_AMT3)", value=2400)
            with gr.Row():
                in_b4 = gr.Number(label="Jun Bill (BILL_AMT4)", value=3000)
                in_b5 = gr.Number(label="May Bill (BILL_AMT5)", value=2800)
                in_b6 = gr.Number(label="Apr Bill (BILL_AMT6)", value=2100)

        with gr.TabItem("3. Debt Repayment History"):
            gr.Markdown("#### Monthly Amount Paid (NT Dollar)")
            with gr.Row():
                in_p1 = gr.Number(label="Sept Paid (PAY_AMT1)", value=2600, minimum=0)
                in_p2 = gr.Number(label="Aug Paid (PAY_AMT2)", value=2400, minimum=0)
                in_p3 = gr.Number(label="Jul Paid (PAY_AMT3)", value=3000, minimum=0)
            with gr.Row():
                in_p4 = gr.Number(label="Jun Paid (PAY_AMT4)", value=2800, minimum=0)
                in_p5 = gr.Number(label="May Paid (PAY_AMT5)", value=2100, minimum=0)
                in_p6 = gr.Number(label="Apr Paid (PAY_AMT6)", value=2500, minimum=0)

    with gr.Row():
        predict_btn = gr.Button("⚡ Run Credit Risk Assessment", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            out_label = gr.Textbox(label="Predicted Credit Risk", interactive=False)
            out_prob = gr.Textbox(label="Default Probability", interactive=False)
        with gr.Column():
            out_details = gr.Markdown()

    all_inputs = [
        in_limit, in_age, in_sex, in_edu, in_marriage,
        in_pay0, in_pay2, in_pay3, in_pay4, in_pay5, in_pay6,
        in_b1, in_b2, in_b3, in_b4, in_b5, in_b6,
        in_p1, in_p2, in_p3, in_p4, in_p5, in_p6,
        demo_truth_state,
    ]

    all_input_components = all_inputs[:-1] + [demo_truth_state]

    load_demo_btn.click(
        fn=populate_demo_profile,
        inputs=[sample_dropdown],
        outputs=all_input_components,
    )

    predict_btn.click(
        fn=predict_credit_risk,
        inputs=all_inputs,
        outputs=[out_label, out_prob, out_details],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
