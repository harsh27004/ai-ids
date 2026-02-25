import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Intrusion Detection SOC",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------
# CYBER THEME STYLING
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #1c1f26;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_components():
    model = joblib.load("unsw_multiclass_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, le = load_components()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🛡️ AI-Based Intrusion Detection System")
st.markdown("### Security Operations Center Dashboard")
st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Network Traffic CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        df["Predicted_Attack"] = le.inverse_transform(predictions)
        df["Confidence (%)"] = np.max(probabilities, axis=1) * 100

        # ---------------------------------------------------
        # SEVERITY MAPPING
        # ---------------------------------------------------
        severity_map = {
            "Normal": "Low",
            "Reconnaissance": "Medium",
            "DoS": "High",
            "Exploits": "Critical",
            "Backdoor": "Critical",
            "Shellcode": "Critical",
            "Worms": "Critical",
            "Generic": "High",
            "Fuzzers": "Medium",
            "Analysis": "Medium"
        }

        df["Severity"] = df["Predicted_Attack"].map(severity_map)

        # ---------------------------------------------------
        # CONFIDENCE FILTER
        # ---------------------------------------------------
        st.markdown("---")
        threshold = st.slider("🔎 Minimum Confidence Level (%)", 0, 100, 50)
        df_filtered = df[df["Confidence (%)"] >= threshold]

        # ---------------------------------------------------
        # SYSTEM STATUS
        # ---------------------------------------------------
        total_attacks = (df_filtered["Predicted_Attack"] != "Normal").sum()

        if total_attacks == 0:
            st.success("✅ SYSTEM STATUS: SECURE - No Intrusions Detected")
        else:
            st.error("🚨 ALERT: Intrusions Detected!")

        # ---------------------------------------------------
        # SUMMARY METRICS
        # ---------------------------------------------------
        st.markdown("## 📊 Threat Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Records", len(df_filtered))
        col2.metric("Normal Traffic", (df_filtered["Predicted_Attack"] == "Normal").sum())
        col3.metric("Detected Attacks", total_attacks)
        col4.metric("Critical Threats",
                    (df_filtered["Severity"] == "Critical").sum())

        st.markdown("---")

        # ---------------------------------------------------
        # ATTACK DISTRIBUTION
        # ---------------------------------------------------
        st.subheader("📊 Attack Distribution")

        attack_counts = df_filtered["Predicted_Attack"].value_counts().reset_index()
        attack_counts.columns = ["Attack Type", "Count"]

        fig_bar = px.bar(
            attack_counts,
            x="Attack Type",
            y="Count",
            color="Attack Type",
            title="Predicted Attack Categories"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            df_filtered,
            names="Predicted_Attack",
            title="Attack Breakdown"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------------------------------------------
        # HIGH RISK ALERTS
        # ---------------------------------------------------
        st.subheader("🚨 High & Critical Risk Alerts")

        high_risk = df_filtered[df_filtered["Severity"].isin(["High", "Critical"])]

        if not high_risk.empty:
            st.dataframe(high_risk)
        else:
            st.info("No high-risk traffic detected.")

        # ---------------------------------------------------
        # FULL RESULTS TABLE
        # ---------------------------------------------------
        st.subheader("📄 All Predictions")
        st.dataframe(df_filtered)

        # ---------------------------------------------------
        # DOWNLOAD OPTION
        # ---------------------------------------------------
        csv = df_filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Detection Report",
            data=csv,
            file_name="IDS_SOC_Report.csv",
            mime="text/csv"
        )

        # ---------------------------------------------------
        # MODEL INFORMATION (FOR VIVA)
        # ---------------------------------------------------
        with st.expander("📘 Model Information"):
            st.write("""
            • Algorithm: Random Forest  
            • Dataset: UNSW-NB15  
            • Multi-class Accuracy: ~76%  
            • Binary Accuracy: ~93%  
            • Class imbalance handled using class_weight  
            • Deployment: Streamlit Dashboard  
            """)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    st.info("Upload a CSV file to begin intrusion detection.")
