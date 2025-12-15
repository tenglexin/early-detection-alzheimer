# 🧠 Alzheimer’s Early Detection System

A machine learning–based web application for early Alzheimer’s disease risk prediction using clinical, demographic, lifestyle, and cognitive data.
The system is powered by a Random Forest classifier and deployed as an interactive Streamlit application.

## 🚀 Project Overview

This project aims to support early risk identification of Alzheimer’s disease by translating a validated machine learning model into a practical decision-support tool.
    
    - Trained and evaluated in Google Colab
    - Best-performing model: Random Forest with SMOTE
    - Deployed using Streamlit and Visual Studio Code
    - Focuses on non-invasive, structured clinical data

> ⚠️ This tool is intended for educational and research purposes only and does not replace professional medical diagnosis.



## 🧠 Model Summary

<img width="947" height="263" alt="image" src="https://github.com/user-attachments/assets/906c7a08-c541-4aed-8c75-7dc319c06994" />


The Random Forest model was selected due to its superior performance and robustness to class imbalance.

## 🖥️ Application Features
- Structured patient input across:
    1. Demographics & lifestyle
    2. Medical history
    3. Clinical measurements
    4. Cognitive & functional assessments
    5. Symptoms

- Probability-based risk prediction with a color-coded risk gauge (Low / Moderate / High)

- Developer Mode for exact-value testing and validation

- End-to-end consistency between training and deployment

## 📂 Project Structure

    ├── app.py                    
    
    ├── models/
    
    │   ├── alzheimers_rf_pipeline.joblib
    
    │   └── feature_names.pkl
    
    ├── requirements.txt
    
    └── README.md

## ▶️ How to Run the App

    1. Install dependencies:
    
        pip install -r requirements.txt
    
    2. Run the Streamlit app:
    
        streamlit run app.py

## 🧪 Developer Mode

The application includes a Developer Mode that allows:
    1. Exact floating-point input values
    2. Replication of test samples from Google Colab
    3. Direct comparison of prediction probabilities

This feature was used to verify identical outputs between offline experiments and real-time deployment, ensuring system reliability and reproducibility.

## 🛠️ Technologies Used

    Python
    Scikit-learn
    Imbalanced-learn (SMOTE)
    Streamlit
    Google Colab
    Visual Studio Code

## 📊 Dataset

Alzheimer’s Disease Dataset (Kaggle)

> https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset 

Structured patient-level clinical, demographic, and cognitive features

## 📌 Disclaimer

This application is a research prototype and should not be used for clinical diagnosis. Predictions are intended to assist early screening and research exploration only.

## 👤 Author

Teng Le Xin

> UTM Master’s Research Project – Alzheimer’s Disease Early Detection using Machine Learning
