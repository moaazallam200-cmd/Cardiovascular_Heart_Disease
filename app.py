# ============================
# Cardiovascular Disease Risk Prediction — Streamlit Clinical Triage App
# ============================

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# Resolve paths whether models are in the root directory or a 'models/' subfolder
current_dir = os.path.dirname(os.path.abspath(__file__))

def resolve_model_path(filename: str) -> str:
    root_candidate = os.path.join(current_dir, filename)
    subfolder_candidate = os.path.join(current_dir, "models", filename)
    return subfolder_candidate if os.path.exists(subfolder_candidate) else root_candidate

@st.cache_resource
def load_models():
    logistic_model = joblib.load(resolve_model_path("logistic_model.pkl"))
    gradient_model = joblib.load(resolve_model_path("gradient_boosting_model.pkl"))
    return logistic_model, gradient_model

logistic_model, gradient_model = load_models()

# ============================
# Header & Clinical Context
# ============================
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown(
    """
    This clinical decision-support tool estimates a patient's **Cardiovascular Disease (CVD)** risk probability
    using leakage-free **HistGradientBoosting** (`0.8003 CV ROC-AUC`) and **Logistic Regression** (`0.7914 CV ROC-AUC`)
    pipelines trained on **68,600+ validated medical examination records**.
    """
)

st.divider()

# ============================
# Patient Clinical Inputs (2-Column Layout)
# ============================
st.subheader("1. Patient Biometrics & Vitals")
col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", min_value=25, max_value=90, value=52, step=1)
    gender_label = st.selectbox("Biological Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165, step=1)
    weight = st.number_input("Weight (kg)", min_value=35.0, max_value=180.0, value=72.0, step=0.5)

with col2:
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi, mmHg)", min_value=80, max_value=240, value=120, step=1)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo, mmHg)", min_value=50, max_value=150, value=80, step=1)
    cholesterol = st.selectbox(
        "Cholesterol Tier",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 — Normal", 2: "2 — Above Normal", 3: "3 — Well Above Normal"}[x]
    )
    gluc = st.selectbox(
        "Glucose Tier",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 — Normal", 2: "2 — Above Normal", 3: "3 — Well Above Normal"}[x]
    )

st.subheader("2. Lifestyle & Screening Sensitivity")
col3, col4, col5 = st.columns(3)
with col3:
    smoke = st.selectbox("Current Smoker", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col4:
    alco = st.selectbox("Alcohol Intake", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
with col5:
    active = st.selectbox("Physically Active", options=[0, 1], value=1, format_func=lambda x: "No" if x == 0 else "Yes")

threshold = st.slider(
    "Clinical Screening Probability Threshold (Lower = Higher Recall / Sensitivity)",
    min_value=0.25, max_value=0.75, value=0.40, step=0.05,
    help="Standard classification uses 0.50. Lowering the threshold to 0.35–0.40 increases Recall (>80%) to minimize missed diagnoses."
)

# ============================
# Derived Features & Validation
# ============================
bmi = weight / ((height / 100.0) ** 2)
pulse_pressure = ap_hi - ap_lo
gender_binary = 0 if gender_label == "Female" else 1

# Show live calculated clinical indices
m1, m2, m3 = st.columns(3)
m1.metric("Calculated BMI", f"{bmi:.1f} kg/m²")
m2.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
m3.metric("BP Reading", f"{ap_hi}/{ap_lo} mmHg")

if ap_hi <= ap_lo:
    st.error("⚠️ Physiological Validation Error: Systolic Blood Pressure (`ap_hi`) must be strictly greater than Diastolic Blood Pressure (`ap_lo`).")
    st.stop()

# Build DataFrame compatible with both the updated 13-feature pipeline and legacy 12-feature pipeline
input_data = pd.DataFrame([{
    "gender": gender_binary,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "pulse_pressure": pulse_pressure,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": smoke,
    "alco": alco,
    "active": active,
    "age_years": int(age_years),
    "bmi": bmi
}])

# ============================
# Prediction & Risk Breakdown
# ============================
if st.button("Evaluate Cardiovascular Risk", type="primary", use_container_width=True):
    gb_proba = float(gradient_model.predict_proba(input_data)[0, 1])
    lr_proba = float(logistic_model.predict_proba(input_data)[0, 1])

    gb_pred = int(gb_proba >= threshold)
    lr_pred = int(lr_proba >= threshold)

    st.divider()
    st.subheader("Clinical Risk Assessment Results")

    res1, res2 = st.columns(2)
    with res1:
        st.markdown("**HistGradientBoosting (Champion Model)**")
        st.metric("CVD Risk Probability", f"{gb_proba * 100:.1f}%")
        if gb_pred == 1:
            st.error(f"⚠️ High Risk (Exceeds {threshold:.2f} threshold)")
        else:
            st.success(f"✅ Low Risk (Below {threshold:.2f} threshold)")

    with res2:
        st.markdown("**Logistic Regression (Baseline)**")
        st.metric("CVD Risk Probability", f"{lr_proba * 100:.1f}%")
        if lr_pred == 1:
            st.error(f"⚠️ High Risk (Exceeds {threshold:.2f} threshold)")
        else:
            st.success(f"✅ Low Risk (Below {threshold:.2f} threshold)")

    if gb_pred == 1 and lr_pred == 1:
        st.error("🚨 **Consensus Triage Recommendation:** Both models flag this profile as **High Cardiovascular Risk**. Recommend secondary diagnostic workup and blood pressure management.")
    elif gb_pred == 0 and lr_pred == 0:
        st.success("✅ **Consensus Triage Recommendation:** Both models classify this profile as **Low Cardiovascular Risk**.")
    else:
        st.warning("⚠️ **Borderline Risk Profile:** The models diverge at the selected threshold. Review the risk probability percentage and clinical blood pressure tier.")
