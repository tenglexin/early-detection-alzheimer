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

# Developer Mode toggle (header-only)
dev_mode = st.checkbox("Developer Mode (Exact Inputs for Spot Testing)")

# Section navigation state
sections = [
    "Demographics & Lifestyle",
    "Medical History",
    "Clinical Measurements",
    "Cognitive",
    "Symptoms"
]
if "current_section" not in st.session_state:
    st.session_state.current_section = 0
if "completed_sections" not in st.session_state:
    st.session_state.completed_sections = {i: False for i in range(len(sections))}


tabs = st.tabs(sections)

# ============================
# Convert all values exactly like Dataset (define before rendering)
# ============================
def convert_to_numeric():
    mapping = {
        "Male": 0, "Female": 1,
        "No": 0, "Yes": 1,
        "Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3,
        "None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3
    }

    data = {
        "Age": st.session_state.get("Age", 70),
        "Gender": mapping[st.session_state.get("Gender", "Male")],
        "Ethnicity": mapping[st.session_state.get("Ethnicity", "Caucasian")],
        "EducationLevel": mapping[st.session_state.get("EducationLevel", "High School")],
        "BMI": st.session_state.get("BMI", 22.0),
        "Smoking": mapping[st.session_state.get("Smoking", "No")],
        "AlcoholConsumption": st.session_state.get("AlcoholConsumption", 2),
        "PhysicalActivity": st.session_state.get("PhysicalActivity", 3 if not dev_mode else 3.0),
        "DietQuality": st.session_state.get("DietQuality", 6 if not dev_mode else 6.0),
        "SleepQuality": st.session_state.get("SleepQuality", 7 if not dev_mode else 7.0),
        "FamilyHistoryAlzheimers": mapping[st.session_state.get("FamilyHistoryAlzheimers", "No")],
        "CardiovascularDisease": mapping[st.session_state.get("CardiovascularDisease", "No")],
        "Diabetes": mapping[st.session_state.get("Diabetes", "No")],
        "Depression": mapping[st.session_state.get("Depression", "No")],
        "HeadInjury": mapping[st.session_state.get("HeadInjury", "No")],
        "Hypertension": mapping[st.session_state.get("Hypertension", "No")],
        "SystolicBP": st.session_state.get("SystolicBP", 120),
        "DiastolicBP": st.session_state.get("DiastolicBP", 80),
        "CholesterolTotal": st.session_state.get("CholesterolTotal", 200),
        "CholesterolLDL": st.session_state.get("CholesterolLDL", 120),
        "CholesterolHDL": st.session_state.get("CholesterolHDL", 55),
        "CholesterolTriglycerides": st.session_state.get("CholesterolTriglycerides", 150),
        "MMSE": st.session_state.get("MMSE", 22 if not dev_mode else 22.0),
        "FunctionalAssessment": st.session_state.get("FunctionalAssessment", 5 if not dev_mode else 5.0),
        "MemoryComplaints": mapping[st.session_state.get("MemoryComplaints", "No")],
        "BehavioralProblems": mapping[st.session_state.get("BehavioralProblems", "No")],
        "ADL": st.session_state.get("ADL", 6 if not dev_mode else 6.0),
        "Confusion": mapping[st.session_state.get("Confusion", "No")],
        "Disorientation": mapping[st.session_state.get("Disorientation", "No")],
        "PersonalityChanges": mapping[st.session_state.get("PersonalityChanges", "No")],
        "DifficultyCompletingTasks": mapping[st.session_state.get("DifficultyCompletingTasks", "No")],
        "Forgetfulness": mapping[st.session_state.get("Forgetfulness", "No")]
    }

    return pd.DataFrame([data])[feature_names]

# Helper: navigation buttons
def nav_controls():
    cols = st.columns(2)
    with cols[0]:
        if st.session_state.current_section > 0:
            if st.button("← Previous", use_container_width=True, key=f"prev_{st.session_state.current_section}"):
                st.session_state.current_section -= 1
                st.rerun()
    with cols[1]:
        if st.session_state.current_section < len(sections) - 1:
            if st.button("Next →", use_container_width=True, key=f"next_{st.session_state.current_section}"):
                st.session_state.completed_sections[st.session_state.current_section] = True
                st.session_state.current_section += 1
                st.rerun()

# Render current section content
with tabs[0]:
    st.subheader("👤 Demographic Details")
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=60, max_value=100, value=70, key="Age")
        Gender = st.selectbox("Gender", ["Male", "Female"], key="Gender")
        Ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"], key="Ethnicity")
        EducationLevel = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"], key="EducationLevel")
    with col2:
        BMI = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=22.0,
            step=0.000001 if dev_mode else 0.1,
            format="%.6f" if dev_mode else None,
            key="BMI"
        )
        Smoking = st.radio("Smoking", ["No", "Yes"], key="Smoking")
        AlcoholConsumption = st.number_input(
            "Alcohol Consumption (per week)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.000001 if dev_mode else 1.0,
            format="%.6f" if dev_mode else None,
            key="AlcoholConsumption"
        )
        PhysicalActivity = st.number_input(
            "Weekly Physical Activity (hours)",
            min_value=0.0 if dev_mode else 0,
            max_value=10.0 if dev_mode else 10,
            value=3.0 if dev_mode else 3,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="PhysicalActivity"
        )
    # No nav_controls needed with tabs

# ============================
# TAB 2: MEDICAL HISTORY
# ============================
with tabs[1]:
    st.subheader("🩺 Medical History")
    mc1, mc2 = st.columns(2)
    with mc1:
        FamilyHistoryAlzheimers = st.radio("Family History of Alzheimer's", ["No", "Yes"], key="FamilyHistoryAlzheimers")
        CardiovascularDisease = st.radio("Cardiovascular Disease", ["No", "Yes"], key="CardiovascularDisease")
        Diabetes = st.radio("Diabetes", ["No", "Yes"], key="Diabetes")
    with mc2:
        Depression = st.radio("Depression", ["No", "Yes"], key="Depression")
        HeadInjury = st.radio("Head Injury", ["No", "Yes"], key="HeadInjury")
        Hypertension = st.radio("Hypertension", ["No", "Yes"], key="Hypertension")
    # No nav_controls needed with tabs

# ============================
# TAB 3: CLINICAL MEASUREMENTS
# ============================
with tabs[2]:
    st.subheader("📊 Clinical Measurements")
    col1, col2 = st.columns(2)
    with col1:
        SystolicBP = st.number_input(
            "Systolic BP",
            min_value=90.0 if dev_mode else 90,
            max_value=180.0 if dev_mode else 180,
            value=120.0 if dev_mode else 120,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="SystolicBP"
        )
        DiastolicBP = st.number_input(
            "Diastolic BP",
            min_value=60.0 if dev_mode else 60,
            max_value=120.0 if dev_mode else 120,
            value=80.0 if dev_mode else 80,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="DiastolicBP"
        )
        CholesterolTotal = st.number_input(
            "Total Cholesterol",
            min_value=150.0 if dev_mode else 150,
            max_value=300.0 if dev_mode else 300,
            value=200.0 if dev_mode else 200,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="CholesterolTotal"
        )
    with col2:
        CholesterolLDL = st.number_input(
            "LDL Cholesterol",
            min_value=50.0 if dev_mode else 50,
            max_value=200.0 if dev_mode else 200,
            value=120.0 if dev_mode else 120,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="CholesterolLDL"
        )
        CholesterolHDL = st.number_input(
            "HDL Cholesterol",
            min_value=20.0 if dev_mode else 20,
            max_value=100.0 if dev_mode else 100,
            value=55.0 if dev_mode else 55,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="CholesterolHDL"
        )
        CholesterolTriglycerides = st.number_input(
            "Triglycerides",
            min_value=50.0 if dev_mode else 50,
            max_value=400.0 if dev_mode else 400,
            value=150.0 if dev_mode else 150,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="CholesterolTriglycerides"
        )

with tabs[3]:
    st.subheader("🧩 Cognitive & Functional Scores")
    col1, col2 = st.columns(2)
    with col1:
        MMSE = st.number_input(
            "MMSE Score",
            min_value=0.0 if dev_mode else 0,
            max_value=30.0 if dev_mode else 30,
            value=22.0 if dev_mode else 22,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="MMSE"
        )
        FunctionalAssessment = st.number_input(
            "Functional Assessment",
            min_value=0.0 if dev_mode else 0,
            max_value=10.0 if dev_mode else 10,
            value=5.0 if dev_mode else 5,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="FunctionalAssessment"
        )
        ADL = st.number_input(
            "ADL Score",
            min_value=0.0 if dev_mode else 0,
            max_value=10.0 if dev_mode else 10,
            value=6.0 if dev_mode else 6,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="ADL"
        )
    with col2:
        DietQuality = st.number_input(
            "Diet Quality Score",
            min_value=0.0 if dev_mode else 0,
            max_value=10.0 if dev_mode else 10,
            value=6.0 if dev_mode else 6,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="DietQuality"
        )
        SleepQuality = st.number_input(
            "Sleep Quality Score",
            min_value=0.0 if dev_mode else 4,
            max_value=10.0 if dev_mode else 10,
            value=7.0 if dev_mode else 7,
            step=0.000001 if dev_mode else 1,
            format="%.6f" if dev_mode else None,
            key="SleepQuality"
        )
    # No nav_controls needed with tabs

if len(sections) > 4:
    with tabs[4]:
        st.subheader("⚠ Symptoms")
        c1, c2, c3 = st.columns(3)
        with c1:
            Confusion = st.radio("Confusion", ["No", "Yes"], key="Confusion")
            Disorientation = st.radio("Disorientation", ["No", "Yes"], key="Disorientation")
            PersonalityChanges = st.radio("Personality Changes", ["No", "Yes"], key="PersonalityChanges")
        with c2:
            DifficultyCompletingTasks = st.radio("Difficulty Completing Tasks", ["No", "Yes"], key="DifficultyCompletingTasks")
            Forgetfulness = st.radio("Forgetfulness", ["No", "Yes"], key="Forgetfulness")
            BehavioralProblems = st.radio("Behavioral Problems", ["No", "Yes"], key="BehavioralProblems")
        with c3:
            MemoryComplaints = st.radio("Memory Complaints", ["No", "Yes"], key="MemoryComplaints")

        # Prediction controls only on last section
        st.write("### Prediction")
        if st.button("🔍 Predict Alzheimer's Risk", use_container_width=True, key="predict_final"):
            input_df = convert_to_numeric()
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # Risk-level gauge
            st.write("### Risk Level")
            # Color-coded risk gauge
            import streamlit.components.v1 as components
            gauge_color = (
                "#27ae60" if prob < 0.33 else
                "#f1c40f" if prob < 0.66 else
                "#e74c3c"
            )
            # Improved risk gauge UI
            gauge_html = f'''
            <div style="width:100%;height:32px;background:#e0e0e0;border-radius:16px;position:relative;margin-bottom:8px;">
                <div style="width:{int(prob*100)}%;height:100%;background:{gauge_color};border-radius:16px;transition:width 0.5s;"></div>
                <div style="position:absolute;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;">
                    <span style="font-size:20px;font-weight:bold;color:#222;text-shadow:0 1px 2px #fff;">{int(prob*100)}%</span>
                </div>
            </div>
            '''
            components.html(gauge_html, height=40)
            # Clear summary below gauge
            if prob < 0.33:
                st.markdown('<div style="color:#27ae60;font-size:18px;font-weight:bold;text-align:center;margin-top:8px;">Low risk: Model predicts a low probability of Alzheimer\'s.</div>', unsafe_allow_html=True)
            elif prob < 0.66:
                st.markdown('<div style="color:#f1c40f;font-size:18px;font-weight:bold;text-align:center;margin-top:8px;">Moderate risk: Model predicts a moderate probability of Alzheimer\'s.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#e74c3c;font-size:18px;font-weight:bold;text-align:center;margin-top:8px;">High risk: Model predicts a high probability of Alzheimer\'s.</div>', unsafe_allow_html=True)