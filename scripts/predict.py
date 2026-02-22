import joblib
import pandas as pd

# Load model
model = joblib.load("unsw_ids_model.pkl")

# Load new sample data
sample = pd.read_csv("sample_input.csv")

# Predict
prediction = model.predict(sample)

print("Prediction:", prediction)
