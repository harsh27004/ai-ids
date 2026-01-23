import streamlit as st
import pandas as pd
from src.predict import predict_from_df

st.set_page_config(page_title="AI Intrusion Detection System", layout="wide")

st.title("🚨 AI-Based Intrusion Detection System")
st.write("Upload network traffic data to detect intrusions.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    df = df.drop(columns=["Label"], errors="ignore")

    if st.button("Run Detection"):
        preds, probs = predict_from_df(df)

        result_df = df.copy()
        result_df["Prediction"] = preds
        result_df["Attack_Probability"] = probs

        st.subheader("Detection Results: ")
        st.dataframe(result_df.head(20))

        st.metric("🚫 Attacks Detected", int(sum(preds)))
        st.metric("✅ Normal Traffic", int(len(preds) - sum(preds)))
