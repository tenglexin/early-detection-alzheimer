import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================
#   LOAD MODEL & FEATURES
# ============================
MODEL_PATH = "models/alzheimers_rf_pipeline.joblib"
FEATURES_PATH = "models/feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Alzheimer's Early Detection",
    page_icon="🧠",
    layout="centered"  # Center the layout for better alignment
)

# ============================
# HEADER
# ============================
st.markdown(
    """
    <style>
        .main-header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }
        .sub-header {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
        }
    </style>
    <div class="main-header">🧠 Alzheimer's Early Detection System</div>
    <div class="sub-header">AI-powered patient assessment</div>
    """,
    unsafe_allow_html=True
)

# ============================
# LAYOUT CONFIGURATION
# ============================
# Use tabs for better organization
st.write("### Patient Information")
tabs = st.tabs(["Demographics & Lifestyle", "Medical History", "Clinical Measurements", "Cognitive & Symptoms"])

# Display the content of the first tab by default
with tabs[0]:
    st.subheader("👤 Demographic Details")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=60, max_value=100, value=70)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        EducationLevel = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"])
    with col2:
        BMI = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
        Smoking = st.radio("Smoking", ["No", "Yes"])
        AlcoholConsumption = st.number_input("Alcohol Consumption (per week)", 0, 20, 2)
        PhysicalActivity = st.number_input("Weekly Physical Activity (hours)", 0, 10, 3)
        DietQuality = st.slider("Diet Quality Score", 0, 10, 6)
        SleepQuality = st.slider("Sleep Quality Score", 4, 10, 7)

# ============================
# TAB 2: MEDICAL HISTORY
# ============================
with tabs[1]:
    st.subheader("🩺 Medical History")
    FamilyHistoryAlzheimers = st.radio("Family History of Alzheimer's", ["No", "Yes"])
    CardiovascularDisease = st.radio("Cardiovascular Disease", ["No", "Yes"])
    Diabetes = st.radio("Diabetes", ["No", "Yes"])
    Depression = st.radio("Depression", ["No", "Yes"])
    HeadInjury = st.radio("Head Injury", ["No", "Yes"])
    Hypertension = st.radio("Hypertension", ["No", "Yes"])

# ============================
# TAB 3: CLINICAL MEASUREMENTS
# ============================
with tabs[2]:
    st.subheader("📊 Clinical Measurements")
    col1, col2 = st.columns(2)
    with col1:
        SystolicBP = st.number_input("Systolic BP", 90, 180, 120)
        DiastolicBP = st.number_input("Diastolic BP", 60, 120, 80)
        CholesterolTotal = st.number_input("Total Cholesterol", 150, 300, 200)
    with col2:
        CholesterolLDL = st.number_input("LDL Cholesterol", 50, 200, 120)
        CholesterolHDL = st.number_input("HDL Cholesterol", 20, 100, 55)
        CholesterolTriglycerides = st.number_input("Triglycerides", 50, 400, 150)

# ============================
# TAB 4: COGNITIVE & SYMPTOMS
# ============================
with tabs[3]:
    st.subheader("🧩 Cognitive & Functional Scores")
    col1, col2 = st.columns(2)
    with col1:
        MMSE = st.slider("MMSE Score", 0, 30, 22)
        FunctionalAssessment = st.slider("Functional Assessment", 0, 10, 5)
        ADL = st.slider("ADL Score", 0, 10, 6)
    with col2:
        Confusion = st.radio("Confusion", ["No", "Yes"])
        Disorientation = st.radio("Disorientation", ["No", "Yes"])
        PersonalityChanges = st.radio("Personality Changes", ["No", "Yes"])
        DifficultyCompletingTasks = st.radio("Difficulty Completing Tasks", ["No", "Yes"])
        Forgetfulness = st.radio("Forgetfulness", ["No", "Yes"])
        BehavioralProblems = st.radio("Behavioral Problems", ["No", "Yes"])
        MemoryComplaints = st.radio("Memory Complaints", ["No", "Yes"])

# ============================
# Convert all values exactly like Dataset
# ============================

def convert_to_numeric():
    mapping = {
        "Male": 0, "Female": 1,
        "No": 0, "Yes": 1,
        "Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3,
        "None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3
    }

    data = {
        "Age": Age,
        "Gender": mapping[Gender],
        "Ethnicity": mapping[Ethnicity],
        "EducationLevel": mapping[EducationLevel],
        "BMI": BMI,
        "Smoking": mapping[Smoking],
        "AlcoholConsumption": AlcoholConsumption,
        "PhysicalActivity": PhysicalActivity,
        "DietQuality": DietQuality,
        "SleepQuality": SleepQuality,
        "FamilyHistoryAlzheimers": mapping[FamilyHistoryAlzheimers],
        "CardiovascularDisease": mapping[CardiovascularDisease],
        "Diabetes": mapping[Diabetes],
        "Depression": mapping[Depression],
        "HeadInjury": mapping[HeadInjury],
        "Hypertension": mapping[Hypertension],
        "SystolicBP": SystolicBP,
        "DiastolicBP": DiastolicBP,
        "CholesterolTotal": CholesterolTotal,
        "CholesterolLDL": CholesterolLDL,
        "CholesterolHDL": CholesterolHDL,
        "CholesterolTriglycerides": CholesterolTriglycerides,
        "MMSE": MMSE,
        "FunctionalAssessment": FunctionalAssessment,
        "MemoryComplaints": mapping[MemoryComplaints],
        "BehavioralProblems": mapping[BehavioralProblems],
        "ADL": ADL,
        "Confusion": mapping[Confusion],
        "Disorientation": mapping[Disorientation],
        "PersonalityChanges": mapping[PersonalityChanges],
        "DifficultyCompletingTasks": mapping[DifficultyCompletingTasks],
        "Forgetfulness": mapping[Forgetfulness]
    }

    return pd.DataFrame([data])[feature_names]


# ============================
# PREDICTION BUTTON
# ============================
st.write("### Prediction")
if st.button("🔍 Predict Alzheimer's Risk", use_container_width=True):
    input_df = convert_to_numeric()
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠ High Risk of Alzheimer's (Probability: {prob:.2f})")
    else:
        st.success(f"✔ Low Risk (Probability: {prob:.2f})")
