import os
from pathlib import Path
import streamlit as st
import pandas as pd
import cloudpickle
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1️⃣ CONFIG & THEME
# ===============================
st.set_page_config(page_title="HR Dashboard - ABC Company", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f7faff;
}
[data-testid="stSidebar"] {
    background-color: #e8f1fa;
    color: black;
}
[data-testid="stSidebar"] * {
    color: #0f4c81 !important;
}
.main-header {
    background-color: #0078d4;
    padding: 12px;
    border-radius: 10px;
    color: white;
    font-size: 26px;
    font-weight: bold;
    text-align: center;
}
h3, h4, h5 {
    color: #0f4c81;
}
div.stButton > button {
    background-color: #0078d4;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: 600;
    border: none;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #0f4c81;
    color: #f0f0f0;
}
[data-testid="stSidebar"] label {
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 2️⃣ LOAD DATA & MODEL
# ===============================
BASE_DIR = Path(os.getcwd())
data_path = BASE_DIR / "data" / "employee_data.csv"
model_path = BASE_DIR / "model2.pkl"

# Load model via cloudpickle
if model_path.exists():
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
else:
    model = None
    st.warning("⚠️ Model file 'model2.pkl' not found.")

# Load dataset
if data_path.exists():
    df = pd.read_csv(data_path)
else:
    st.warning("⚠️ Data not found. Please upload 'employee_data.csv' in the /data folder.")
    df = pd.DataFrame()

# Kolom wajib
required_cols = [
    "Age", "Years_at_Company", "Performance_Score", "Leadership_Score",
    "Training_Hours", "Projects_Handled", "Peer_Review_Score",
    "Current_Position_Level", "Promotion_Eligible"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Missing columns: {missing_cols}")
else:
    df = df.dropna(subset=["Projects_Handled", "Current_Position_Level"], how="any")

# ===============================
# 3️⃣ HEADER
# ===============================
st.markdown('<div class="main-header">HR Dashboard - ABC Company</div>', unsafe_allow_html=True)
st.markdown("---")

# ===============================
# 4️⃣ SIDEBAR NAVIGATION (KIRI)
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"],
    index=2
)

# ===============================
# 5️⃣ PAGE: GENERAL DASHBOARD
# ===============================
if menu == "General Dashboard":
    st.subheader("General Dashboard")
    st.info("Welcome to the HR Analytics Dashboard. Please select another tab to explore insights.")

# ===============================
# 6️⃣ PAGE: TALENT DEVELOPMENT
# ===============================
elif menu == "Talent Development":
    st.subheader("Talent Development")
    st.info("This section will display training and development analytics (under construction).")

# ===============================
# 7️⃣ PAGE: ABSENCY
# ===============================
elif menu == "Absency":
    st.subheader("Absency Overview")
    st.info("Absency data visualization coming soon.")

# ===============================
# 8️⃣ PAGE: RECRUITMENT
# ===============================
elif menu == "Recruitment":
    st.subheader("Recruitment Analytics")
    st.info("Recruitment insights and metrics coming soon.")

# ===============================
# 9️⃣ PAGE: PROMOTION
# ===============================
elif menu == "Promotion":
    st.subheader("Promotion Dashboard")

    if not df.empty:
        # ===== Boxplot =====
        st.markdown("### Projects Handled Distribution by Seniority Level")
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Current_Position_Level",
            y="Projects_Handled",
            hue="Promotion_Eligible",
            palette=["#ff7f0e", "#1f77b4"]
        )
        plt.xlabel("Seniority Level")
        plt.ylabel("Projects Handled")
        plt.legend(title="Promotion Eligibility", loc="upper right")
        st.pyplot(plt)

        # ===== Predictive Table =====
        st.markdown("---")
        st.subheader("Predictive HR - Top 5 Recommended for Promotion")
        if model:
            X = df.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
            df["Predicted_Promotion"] = model.predict(X)
            top5 = df[df["Predicted_Promotion"] == 1].head(5)
            st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Years_at_Company"]])
        else:
            st.info("Upload model2.pkl to enable prediction.")
    else:
        st.info("Upload dataset to view promotion dashboard.")

    # ===============================
    # 🔐 SIDEBAR KANAN (ADMIN INPUT)
    # ===============================
    with st.sidebar:
        st.markdown("---")
        st.subheader("Data Input for Employee Promotion Prediction")

        password = st.text_input("Enter Admin Password", type="password")

        if password == "0000":
            st.success("Access Granted ✅")

            emp_id = st.text_input("Employee ID (Format: EMPXXXX)")
            position = st.selectbox("Position Level", df["Current_Position_Level"].unique() if "Current_Position_Level" in df.columns else [])
            perf_score = st.number_input("Performance Score (1–5)", min_value=1, max_value=5, step=1)
            years = st.number_input("Years at Company", min_value=0, max_value=50, step=1)

            if st.button("Predict Promotion"):
                if model:
                    new_X = pd.DataFrame({
                        "Years_at_Company": [years],
                        "Performance_Score": [perf_score],
                        "Current_Position_Level": [position]
                    })
                    pred = model.predict(new_X)[0]
                    st.success(f"{emp_id}: {'✅ Eligible' if pred==1 else 'Not Eligible'}")
                else:
                    st.error("⚠️ Model not loaded. Please upload model2.pkl.")
        else:
            st.info("Enter admin password to unlock input features.")
