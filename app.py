import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# Configure the page layout
st.set_page_config(page_title="Cardiovascular Disease Prediction", page_icon="❤️", layout="wide")

# Get the absolute path to the directory where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute paths to the pickle files
logistic_path = os.path.join(current_dir, "logistic_model.pkl")
gradient_path = os.path.join(current_dir, "gradient_boosting_model.pkl")

# Load Models
@st.cache_resource
def load_models():
    log_model = joblib.load(logistic_path)
    grad_model = joblib.load(gradient_path)
    return log_model, grad_model

logistic_model, gradient_model = load_models()

# Function to create a confidence gauge chart
def create_gauge_chart(probability, model_name):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 24}},
        title={'text': f"{model_name}<br>High Risk Confidence", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#e74c3c" if probability >= 0.5 else "#2ecc71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(46, 204, 113, 0.3)"},
                {'range': [50, 100], 'color': "rgba(231, 76, 60, 0.3)"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Project Title & Description
st.title("❤️ Cardiovascular Disease Risk Prediction")
st.markdown("""
This application utilizes Machine Learning to predict the risk of cardiovascular disease based on standard medical examination data. 
Please enter the patient's details below to generate a risk assessment and view the model confidence metrics.
""")
st.markdown("---")

# User Input - Grid Layout
st.subheader("Patient Medical Information")
col1, col2, col3 = st.columns(3)

with col1:
    age_years = st.number_input("Age (Years)", min_value=18, max_value=120, value=50, step=1)
    gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

with col2:
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=50, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])
    gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1])

with col3:
    smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.markdown("---")

# Prediction Execution
if st.button("Generate Risk Assessment", type="primary", use_container_width=True):
    
    # Calculate BMI
    height_meters = height / 100.0
    bmi = weight / (height_meters ** 2)

    # Structure the dataframe exactly as expected by the pipelines
    input_data = pd.DataFrame([[
        gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, bmi
    ]], columns=[
        "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "age_years", "bmi"
    ])

    # Predictions & Probabilities
    log_pred = logistic_model.predict(input_data)[0]
    log_proba = logistic_model.predict_proba(input_data)[0][1] # Probability of Class 1 (High Risk)
    
    grad_pred = gradient_model.predict(input_data)[0]
    grad_proba = gradient_model.predict_proba(input_data)[0][1] # Probability of Class 1 (High Risk)

    # Display Dashboard
    st.subheader("Model Confidence Dashboard")
    
    dash_col1, dash_col2 = st.columns(2)
    
    with dash_col1:
        if log_pred == 1:
            st.error("### Logistic Regression: ⚠️ High Risk")
        else:
            st.success("### Logistic Regression: ✅ Low Risk")
        st.plotly_chart(create_gauge_chart(log_proba, "Logistic Regression"), use_container_width=True)

    with dash_col2:
        if grad_pred == 1:
            st.error("### Gradient Boosting: ⚠️ High Risk")
        else:
            st.success("### Gradient Boosting: ✅ Low Risk")
        st.plotly_chart(create_gauge_chart(grad_proba, "Gradient Boosting"), use_container_width=True)

    # Final Consensus
    st.markdown("### Consensus Assessment")
    if log_pred == grad_pred:
        if log_pred == 1:
            st.error("🚨 Both models indicate a **High Risk** of Cardiovascular Disease.")
        else:
            st.success("✅ Both models indicate a **Low Risk** of Cardiovascular Disease.")
    else:
        st.warning("⚠️ The models provided conflicting predictions. Please review the confidence distributions above.")