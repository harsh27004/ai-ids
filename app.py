import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
html, body, [class*="css"]  {
    background-color: #0b0f19;
    color: #e6f1ff;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00f5ff;
    text-shadow: 0 0 15px #00f5ff;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.metric-card {
    background: linear-gradient(145deg, #111827, #1f2937);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    transition: 0.3s;
}

.metric-card:hover {
    box-shadow: 0 0 25px rgba(0,255,255,0.5);
    transform: scale(1.03);
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    color: #00f5ff;
}

.severity-low {color: #22c55e; font-weight: bold;}
.severity-medium {color: #facc15; font-weight: bold;}
.severity-high {color: #f97316; font-weight: bold;}
.severity-critical {color: #ef4444; font-weight: bold;}
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
st.markdown('<div class="main-title">🛡 AI-Powered Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">UNSW-NB15 | Security Operations Center Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Network Traffic CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

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
        st.markdown("### 🔎 Confidence Filter")
        threshold = st.slider("Minimum Confidence Level (%)", 0, 100, 50)
        df_filtered = df[df["Confidence (%)"] >= threshold]

        total_records = len(df_filtered)
        total_attacks = (df_filtered["Predicted_Attack"] != "Normal").sum()
        critical_count = (df_filtered["Severity"] == "Critical").sum()
        normal_count = (df_filtered["Predicted_Attack"] == "Normal").sum()

        attack_percentage = (total_attacks / total_records) * 100 if total_records > 0 else 0

        # ---------------------------------------------------
        # SYSTEM STATUS
        # ---------------------------------------------------
        st.markdown("## 🚦 System Status")

        if attack_percentage < 20:
            st.success("🟢 System Secure - Low Threat Activity")
        elif attack_percentage < 50:
            st.warning("🟡 Moderate Risk Detected")
        else:
            st.error("🔴 High Threat Level - Immediate Attention Required")

        # ---------------------------------------------------
        # METRIC CARDS
        # ---------------------------------------------------
        st.markdown("## 📊 Threat Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="metric-card"><div>Total Records</div><div class="metric-value">{total_records}</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="metric-card"><div>Normal Traffic</div><div class="metric-value">{normal_count}</div></div>', unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="metric-card"><div>Detected Attacks</div><div class="metric-value">{total_attacks}</div></div>', unsafe_allow_html=True)

        with col4:
            st.markdown(f'<div class="metric-card"><div>Critical Threats</div><div class="metric-value">{critical_count}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------
        # RISK GAUGE
        # ---------------------------------------------------
        st.subheader("📈 Threat Level Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attack_percentage,
            title={'text': "Attack Percentage"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00f5ff"},
                'steps': [
                    {'range': [0, 20], 'color': "#14532d"},
                    {'range': [20, 50], 'color': "#78350f"},
                    {'range': [50, 100], 'color': "#7f1d1d"},
                ],
            }
        ))

        st.plotly_chart(fig_gauge, use_container_width=True)

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
            template="plotly_dark"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(
            df_filtered,
            names="Predicted_Attack",
            template="plotly_dark"
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------------------------------------------
        # HIGH RISK ALERT TABLE
        # ---------------------------------------------------
        st.subheader("🚨 High & Critical Risk Traffic")

        high_risk = df_filtered[df_filtered["Severity"].isin(["High", "Critical"])]

        if not high_risk.empty:
            st.dataframe(high_risk, use_container_width=True)
        else:
            st.info("No high-risk traffic detected.")

        # ---------------------------------------------------
        # DOWNLOAD REPORT
        # ---------------------------------------------------
        csv = df_filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Full Detection Report",
            data=csv,
            file_name="AI_IDS_SOC_Report.csv",
            mime="text/csv"
        )

        # ---------------------------------------------------
        # MODEL INFO (VIVA READY)
        # ---------------------------------------------------
        with st.expander("📘 Model & System Information"):
            st.write("""
            **Model:** Random Forest Classifier  
            **Dataset:** UNSW-NB15  
            **Binary Accuracy:** ~93%  
            **Multi-class Accuracy:** ~76%  
            **Feature Engineering:** StandardScaler + OneHotEncoder  
            **Deployment:** Streamlit SOC Dashboard  
            """)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

else:
    st.info("⬆ Upload a CSV file to start intrusion detection analysis.")
