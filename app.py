import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CYBER THEME CSS
# ---------------------------------------------------
st.markdown("""
<style>

body {
background: linear-gradient(270deg,#0b0f19,#111827,#0b0f19);
background-size: 400% 400%;
animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
0% {background-position:0% 50%;}
50% {background-position:100% 50%;}
100% {background-position:0% 50%;}
}

[data-testid="stSidebar"] {
background-color:#0f172a;
width:350px;
}

.metric-card {
background:#111827;
padding:25px;
border-radius:15px;
text-align:center;
transition:0.3s;
box-shadow:0 0 15px rgba(0,255,255,0.2);
}

.metric-card:hover {
transform:scale(1.05);
box-shadow:0 0 30px rgba(0,255,255,0.6);
}

.metric-value {
font-size:32px;
font-weight:bold;
color:#00f5ff;
}

.section-title {
color:#00f5ff;
text-shadow:0 0 15px #00f5ff;
}

[data-testid="stFileUploader"]{
border:2px dashed #00f5ff;
border-radius:15px;
padding:20px;
background:#0f172a;
box-shadow:0 0 20px rgba(0,255,255,0.3);
width:60%;
margin:auto;
}

.counter-box{
background:#0f172a;
padding:20px;
border-radius:12px;
text-align:center;
font-size:28px;
color:#ff4b4b;
box-shadow:0 0 15px rgba(255,0,0,0.4);
animation:pulse 2s infinite;
}

@keyframes pulse{
0%{opacity:1}
50%{opacity:0.6}
100%{opacity:1}
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
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Welcome...")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview","Threat Analytics","Traffic Explorer","Model Info"]
)

threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    0,100,50
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("<h1 class='section-title'>AI-Powered Intrusion Detection System</h1>",unsafe_allow_html=True)
st.caption("Security Operations Center Dashboard")

# ---------------------------------------------------
# FILE UPLOAD CENTER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    with st.spinner("Analyzing network traffic..."):

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

    df["Predicted_Attack"] = le.inverse_transform(predictions)
    df["Confidence (%)"] = np.max(probabilities, axis=1) * 100

    severity_map = {
        "Normal":"Low",
        "Reconnaissance":"Medium",
        "DoS":"High",
        "Exploits":"Critical",
        "Backdoor":"Critical",
        "Shellcode":"Critical",
        "Worms":"Critical",
        "Generic":"High",
        "Fuzzers":"Medium",
        "Analysis":"Medium"
    }

    df["Severity"] = df["Predicted_Attack"].map(severity_map)

    df_filtered = df[df["Confidence (%)"] >= threshold]

    total_records = len(df_filtered)
    total_attacks = (df_filtered["Predicted_Attack"] != "Normal").sum()
    critical_count = (df_filtered["Severity"]=="Critical").sum()
    normal_count = (df_filtered["Predicted_Attack"]=="Normal").sum()

    attack_percentage = (total_attacks/total_records)*100 if total_records else 0

# ---------------------------------------------------
# ALERT COUNTER
# ---------------------------------------------------
    st.markdown(f"""
    <div class="counter-box">
    🚨 Active Threats Detected: {total_attacks}
    </div>
    """,unsafe_allow_html=True)

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------
    if menu == "Overview":

        st.markdown("<h2 class='section-title'>Threat Summary</h2>",unsafe_allow_html=True)

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.markdown(f'<div class="metric-card">Total Records<div class="metric-value">{total_records}</div></div>',unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="metric-card">Normal Traffic<div class="metric-value">{normal_count}</div></div>',unsafe_allow_html=True)

        with col3:
            st.markdown(f'<div class="metric-card">Detected Attacks<div class="metric-value">{total_attacks}</div></div>',unsafe_allow_html=True)

        with col4:
            st.markdown(f'<div class="metric-card">Critical Threats<div class="metric-value">{critical_count}</div></div>',unsafe_allow_html=True)

        st.markdown("---")

        left,right = st.columns(2)

        with left:

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=attack_percentage,
                title={'text':"Threat Level (%)"},
                gauge={
                    'axis':{'range':[0,100]},
                    'bar':{'color':"#00f5ff"}
                }
            ))

            fig_gauge.update_layout(
                paper_bgcolor="#0b0f19",
                font={'color':"#00f5ff"}
            )

            st.plotly_chart(fig_gauge,use_container_width=True)

        with right:

            fig_pie = px.pie(
                df_filtered,
                names="Predicted_Attack",
                template="plotly_dark"
            )

            st.plotly_chart(fig_pie,use_container_width=True)

# ---------------------------------------------------
# THREAT ANALYTICS
# ---------------------------------------------------
    elif menu == "Threat Analytics":

        st.markdown("<h2 class='section-title'>Attack Distribution</h2>",unsafe_allow_html=True)

        attack_counts = df_filtered["Predicted_Attack"].value_counts().reset_index()
        attack_counts.columns = ["Attack Type","Count"]

        fig_bar = px.bar(
            attack_counts,
            x="Attack Type",
            y="Count",
            color="Attack Type",
            template="plotly_dark"
        )

        st.plotly_chart(fig_bar,use_container_width=True)

        # Severity Distribution
        st.subheader("Threat Severity Distribution")

        severity_counts = df_filtered["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity","Count"]

        fig_severity = px.bar(
            severity_counts,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map={
                "Low":"green",
                "Medium":"orange",
                "High":"red",
                "Critical":"purple"
            },
            template="plotly_dark"
        )

        st.plotly_chart(fig_severity,use_container_width=True)

        # Heatmap
        st.subheader("Attack Heatmap")

        pivot = pd.pivot_table(
            df_filtered,
            values="Confidence (%)",
            index="Predicted_Attack",
            columns="Severity",
            aggfunc="count",
            fill_value=0
        )

        fig_heatmap = px.imshow(pivot,
                                text_auto=True,
                                color_continuous_scale="Turbo",
                                template="plotly_dark")

        st.plotly_chart(fig_heatmap,use_container_width=True)

# ---------------------------------------------------
# TRAFFIC EXPLORER
# ---------------------------------------------------
    elif menu == "Traffic Explorer":

        st.markdown("<h2 class='section-title'>Network Traffic Explorer</h2>",unsafe_allow_html=True)

        st.dataframe(df_filtered,use_container_width=True)

        csv = df_filtered.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Detection Report",
            csv,
            "AI_IDS_Report.csv",
            "text/csv"
        )

# ---------------------------------------------------
# MODEL INFO
# ---------------------------------------------------
    elif menu == "Model Info":

        st.markdown("<h2 class='section-title'>Model Information</h2>",unsafe_allow_html=True)

        st.markdown("""
**Model:** Random Forest  
**Dataset:** UNSW-NB15  

Binary Accuracy: ~93%  
Multi-class Accuracy: ~76%
""")

        st.subheader("Feature Importance")

        try:

            importances = model.feature_importances_
            features = df.columns[:len(importances)]

            importance_df = pd.DataFrame({
                "Feature":features,
                "Importance":importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig_imp = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                template="plotly_dark"
            )

            st.plotly_chart(fig_imp,use_container_width=True)

        except:
            st.warning("Feature importance unavailable")

        # Confusion Matrix
        if "label" in df.columns:

            st.subheader("Confusion Matrix")

            y_true = df["label"]
            y_pred = df["Predicted_Attack"]

            cm = confusion_matrix(y_true,y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm,annot=True,fmt="d",cmap="coolwarm",ax=ax)

            st.pyplot(fig)

else:

    st.info("Upload a CSV file above to begin intrusion detection analysis.")
