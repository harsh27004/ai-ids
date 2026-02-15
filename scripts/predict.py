import joblib
import pandas as pd
import numpy as np
import json

# Load model
model = joblib.load("models/random_forest_ids.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load feature columns
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

def predict_sample(sample_dict):
    df = pd.DataFrame([sample_dict])
    df = df[feature_columns]
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    return "Attack" if prediction == 1 else "Benign"

print("Model Ready for Inference")
