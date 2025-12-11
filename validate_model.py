import pandas as pd
import joblib

model = joblib.load("models/alzheimers_rf_pipeline.joblib")
feature_names = joblib.load("models/feature_names.pkl")

# Updated the file path to point to the correct location of the test set
df = pd.read_csv("models/test_set_for_validation.csv")

X = df[feature_names]
y = df["Diagnosis"]

pred = model.predict(X)
proba = model.predict_proba(X)[:,1]

from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y, pred))
print(classification_report(y, pred))
