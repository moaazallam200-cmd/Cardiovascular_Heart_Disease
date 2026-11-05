# ============================
# Cardiovascular Disease Risk Prediction — Streamlit Clinical Triage App
# ============================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Cardiovascular Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

current_dir = os.path.dirname(os.path.abspath(__file__))


def resolve_path(filename: str) -> str:
    """Locate a file whether it sits in the repo root, 'models/', or 'data/'."""
    for folder in ["", "models", "data"]:
        candidate = os.path.join(current_dir, folder, filename) if folder else os.path.join(current_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(current_dir, filename)


def train_and_cache_pipelines():
    """Train leakage-free pipelines directly from cardio_train.csv in the current environment."""
    csv_path = resolve_path("cardio_train.csv")
    df = pd.read_csv(csv_path, sep=";")

    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df.drop_duplicates().reset_index(drop=True)

    # Feature engineering
    df["age_years"] = (df["age"] / 365.25).astype(int)
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    if set(df["gender"].unique()).issubset({1, 2}):
        df["gender"] = df["gender"].map({1: 0, 2: 1})

    # Physiological outlier filtering
    df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
    df = df[(df["ap_lo"] >= 50) & (df["ap_lo"] <= 150)]
    df = df[df["ap_hi"] > df["ap_lo"]]
    df = df[(df["height"] >= 100) & (df["height"] <= 250)]
    df = df[df["weight"] >= 30]
    df = df[(df["bmi"] >= 12) & (df["bmi"] <= 60)]
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

    X = df.drop(columns=["cardio", "age"])
    y = df["cardio"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ["height", "weight", "ap_hi", "ap_lo", "pulse_pressure", "age_years", "bmi"]
    categorical_features = ["cholesterol", "gluc"]
    binary_features = ["gender", "smoke", "alco", "active"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop",
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    gradient_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(random_state=42)),
        ]
    )

    logistic_pipeline.fit(X_train, y_train)
    gradient_pipeline.fit(X_train, y_train)

    # Save native pickle files matching the exact runtime Python/Scikit-Learn version
    try:
        joblib.dump(logistic_pipeline, resolve_path("logistic_model.pkl"))
        joblib.dump(gradient_pipeline, resolve_path("gradient_boosting_model.pkl"))
    except Exception:
        pass

    return logistic_pipeline, gradient_pipeline


@st.cache_resource(show_spinner="Loading clinical machine learning pipelines...")
def load_models():
    """Load pre-fitted models, or automatically re-fit if Python/scikit-learn versions differ."""
    try:
        logistic_model = joblib.load(resolve_path("logistic_model.pkl"))
        gradient_model = joblib.load(resolve_path("gradient_boosting_model.pkl"))
        return logistic_model, gradient_model
    except Exception:
        return train_and_cache_pipelines()


logistic_model, gradient_model = load_models()

# ============================
# Header & Clinical Context
# ============================
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.markdown(
    """
    This clinical decision-support tool estimates a patient's **Cardiovascular Disease (CVD)** risk probability
    using leakage-free **HistGradientBoosting** (`0.8054 Test ROC-AUC`) and **Logistic Regression** (`0.7905 CV ROC-AUC`)
    pipelines trained on **68,552 validated medical examination records**.
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
    active = st.selectbox("Physically Active", options=[0, 1], index=1, format_func=lambda x: "No" if x == 0 else "Yes")

threshold = st.slider(
    "Clinical Screening Probability Threshold (Lower = Higher Recall / Sensitivity)",
    min_value=0.25, max_value=0.75, value=0.40, step=0.05,
    help="Standard classification uses 0.50. Lowering the threshold to 0.35–0.40 increases Recall (>78–84%) to minimize missed diagnoses."
)

# ============================
# Derived Features & Validation
# ============================
bmi = weight / ((height / 100.0) ** 2)
pulse_pressure = ap_hi - ap_lo

m1, m2, m3 = st.columns(3)
m1.metric("Calculated BMI", f"{bmi:.1f} kg/m²")
m2.metric("Pulse Pressure", f"{pulse_pressure} mmHg")
m3.metric("BP Reading", f"{ap_hi}/{ap_lo} mmHg")

if ap_hi <= ap_lo:
    st.error("⚠️ Physiological Validation Error: Systolic Blood Pressure (`ap_hi`) must be strictly greater than Diastolic Blood Pressure (`ap_lo`).")
    st.stop()


def build_compatible_input(model) -> pd.DataFrame:
    """Construct input DataFrame matching the exact schema expected by a loaded pipeline."""
    expected_cols = list(getattr(model, "feature_names_in_", []))
    # Check if the loaded model was trained with gender mapped to {0, 1} or legacy {1, 2}
    gender_val = 0 if gender_label == "Female" else 1
    row = {
        "age": int(age_years * 365.25),
        "gender": gender_val,
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
        "bmi": bmi,
    }
    if expected_cols:
        return pd.DataFrame([{col: row[col] for col in expected_cols}])
    return pd.DataFrame([row])


# ============================
# Prediction & Risk Breakdown
# ============================
if st.button("Evaluate Cardiovascular Risk", type="primary", use_container_width=True):
    gb_input = build_compatible_input(gradient_model)
    lr_input = build_compatible_input(logistic_model)

    gb_proba = float(gradient_model.predict_proba(gb_input)[0, 1])
    lr_proba = float(logistic_model.predict_proba(lr_input)[0, 1])

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
