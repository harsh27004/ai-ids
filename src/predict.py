import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Load model & scaler once
ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models/final_rf_model.joblib")
scaler = joblib.load(ROOT / "models/standard_scaler.joblib")

def predict_from_df(df: pd.DataFrame):
    """
    Input: df with same structure as training X
    Output:
      preds: 0/1 predictions
      probs: probability of attack (class 1)
    """

    df = df.copy()

    # DROP non-numeric columns (Timestamp etc.)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        df = df.drop(columns=non_numeric_cols)

    # Safety check
    if df.isnull().any().any():
        raise ValueError("Input contains NaN values")

    X = df.astype(float) 
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    return preds, probs
