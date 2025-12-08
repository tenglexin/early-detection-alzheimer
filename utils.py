# utils.py
import pandas as pd
import numpy as np

def make_input_df_from_form(feature_names, form_values):
    """
    feature_names: list of column names expected by pipeline
    form_values: dict mapping name->value from Streamlit inputs
    Returns: single-row DataFrame with columns in correct order
    """
    row = {c: form_values.get(c, np.nan) for c in feature_names}
    return pd.DataFrame([row], columns=feature_names)

def batch_predict_from_file(pipeline, uploaded_df, feature_names):
    """
    pipeline: loaded preproc+clf
    uploaded_df: DataFrame from uploaded CSV
    feature_names: list of expected features
    Returns: DataFrame with predictions and probabilities appended
    """
    # Keep only the expected features (and warn if missing)
    missing = [c for c in feature_names if c not in uploaded_df.columns]
    if missing:
        raise ValueError(f"Missing columns in uploaded file: {missing}")
    X = uploaded_df[feature_names].copy()
    preds = pipeline.predict(X)
    probas = None
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)[:, 1]  # positive class prob
    res = uploaded_df.copy()
    res['predicted_label'] = preds
    if probas is not None:
        res['predicted_prob'] = probas
    return res