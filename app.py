# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from utils import make_input_df_from_form, batch_predict_from_file
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(page_title="Alzheimer's Early Detection Demo", layout="centered")

# --- Config: path to your saved pipeline ---
MODEL_PATH = "models/alzheimers_rf_pipeline.joblib"  # update if different

st.title("Early Detection of Alzheimer’s Disease — Demo")
st.markdown("""
This demo uses a saved Random Forest pipeline (preprocessing + classifier) to predict Alzheimer's diagnosis.
Upload a CSV for batch predictions or input features for a single prediction.
""")

# Load model
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Place model in models/ and restart.")
        return None
    pipeline = joblib.load(path)
    return pipeline

pipeline = load_model(MODEL_PATH)
if pipeline is None:
    st.stop()

# Determine feature names expected by pipeline (from preprocessor)
# We try to infer expected feature names from pipeline preprocessor if possible.
try:
    preproc = pipeline.named_steps['preproc']
    # if ColumnTransformer with transformers, try to assemble feature names
    col_names = None
    if hasattr(preproc, "transformers_"):
        # best-effort to reconstruct numeric and categorical names from fitted transformers:
        # In many cases getting feature names is nontrivial; fallback to reading from a saved list if you stored it.
        # For simplicity, require user to upload CSV with appropriate columns or read from sample header.
        col_names = None
    else:
        col_names = None
except Exception:
    col_names = None

st.sidebar.header("Options")
show_metrics = st.sidebar.checkbox("Show evaluation metrics (when uploading labelled test CSV)", value=True)
download_sample = st.sidebar.checkbox("Show CSV sample for batch input", value=True)

st.header("Single sample prediction")
st.write("Enter feature values for a single patient.")

# For simplicity: attempt to get feature list from a sample CSV (you can replace this with a hard-coded list)
# If you want a hard-coded feature list, put it here:
FEATURES = None
if FEATURES is None:
    # try to read from an example file shipped with app or allow user to upload sample
    st.info("If the app can't auto-detect features, upload a sample CSV or paste values for the fields.")
    uploaded_sample = st.file_uploader("(Optional) Upload a sample CSV with column headers to auto-fill the form", type=["csv"], accept_multiple_files=False)
    if uploaded_sample is not None:
        df_sample = pd.read_csv(uploaded_sample)
        FEATURES = [c for c in df_sample.columns if c != 'Diagnosis' and c != 'DoctorInCharge' and c != 'PatientID']
    else:
        # fallback: ask user to upload at least once or manually specify
        if st.button("I will provide a sample CSV later; show manual input form"):
            st.info("Upload a CSV in the 'Batch prediction' section and use 'Single sample' later.")
        FEATURES = []

# If we have features, render inputs
form_values = {}
if FEATURES:
    with st.form(key="single_form"):
        for col in FEATURES:
            # assume numeric for your dataset — use number_input
            default = 0.0
            try:
                # If sample df exists, infer mean
                if 'df_sample' in locals() and col in df_sample.columns:
                    default = float(df_sample[col].mean())
            except Exception:
                default = 0.0
            form_values[col] = st.number_input(f"{col}", value=default, format="%.6f")
        submit = st.form_submit_button("Predict single sample")
    if submit:
        X_single = make_input_df_from_form(FEATURES, form_values)
        pred = pipeline.predict(X_single)[0]
        st.success(f"Predicted label: {pred}")
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X_single)[:, 1][0]
            st.write(f"Probability (positive class): {prob:.3f}")

else:
    st.warning("No feature list detected. Upload a sample CSV in 'Batch prediction' to enable single-sample form.")

st.header("Batch prediction (CSV)")
uploaded_file = st.file_uploader("Upload CSV file with the same columns used during model training", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        # remove non-feature columns if present
        candidate_features = [c for c in df_uploaded.columns if c not in ('PatientID', 'DoctorInCharge', 'Diagnosis')]
        st.write("Uploaded file columns:", df_uploaded.columns.tolist())
        st.write("Using features:", candidate_features)
        results_df = batch_predict_from_file(pipeline, df_uploaded, candidate_features)
        st.write("Predictions (first 10 rows):")
        st.dataframe(results_df.head(10))
        # Provide download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
        # If label column present and user wants metrics
        if 'Diagnosis' in df_uploaded.columns and show_metrics:
            y_true = df_uploaded['Diagnosis']
            y_pred = results_df['predicted_label']
            st.subheader("Evaluation metrics on uploaded labelled file")
            st.text("Confusion Matrix:")
            st.write(confusion_matrix(y_true, y_pred))
            st.text("Classification report:")
            st.text(classification_report(y_true, y_pred))
            if hasattr(pipeline, "predict_proba"):
                try:
                    roc = roc_auc_score(y_true, pipeline.predict_proba(df_uploaded[candidate_features])[:,1])
                    st.write("ROC-AUC:", roc)
                except Exception as e:
                    st.write("ROC-AUC could not be computed:", e)
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")

st.markdown("---")
st.markdown("**How to use**: \n1. Prepare CSV with columns matching training features (exclude PatientID/DoctorInCharge/Diagnosis for batch prediction if you don't want labels). \n2. For single sample, upload a sample CSV first to auto-fill form OR hard-code FEATURES list in app.py. \n3. Click Predict and download results.")

st.sidebar.markdown("### Deployment tips")
st.sidebar.info("To run locally: `streamlit run app.py`  \nTo deploy: push repo to Streamlit Cloud and point to this app. Make sure models/ contains your joblib file.")