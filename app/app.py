# app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# 1. CONFIG & THEME
# ===============================
st.set_page_config(
    page_title="HR Dashboard - ABC Company",
    layout="wide"
)

# Custom CSS for modern blue theme
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #f7faff;
        }
        [data-testid="stSidebar"] {
            background-color: #0f4c81;
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
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
        h3 {
            color: #0f4c81;
        }
        div[data-testid="stMetricValue"] {
            color: #0078d4;
        }
        div.stButton > button {
            background-color: #0078d4;
            color: white;
            border-radius: 8px;
            height: 3em;
            font-weight: 600;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #0f4c81;
            color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 2️. LOAD DATA / MODEL
# ===============================
# Path absolut ke folder tempat app.py berada
if "__file__" in globals():
    BASE_DIR = Path(__file__).parent
else:
    BASE_DIR = Path(os.getcwd())

# Path aman
data_path = BASE_DIR.parent / "Data" / "Rakamin Bootcamp - Dataset - Promotion Dataset.csv"
model_path = BASE_DIR / "model.pkl"

# Load data dan model
df = pd.read_csv(data_path, sep=";")
model = joblib.load(model_path)

# ===============================
# 3️. HEADER
# ===============================
# Path absolut file logo relatif terhadap lokasi app.py
logo_path = Path(__file__).parent / "ALGORANGER 2 Logo with Graph and Hat (1).png"

col1, col2 = st.columns([1, 5])
with col1:
    st.image(str(logo_path), width=85)
with col2:
    st.markdown('<div class="main-header">HR Dashboard - ABC Company</div>', unsafe_allow_html=True)

st.markdown("---")

# ===============================
# 4️. SIDEBAR NAVIGATION
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"],
    index=2
)

# ===============================
# 5️. TAB: PROMOTION DASHBOARD
# ===============================
if menu == "Promotion":
    st.subheader("HR - Promotion Dashboard")

    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        avg_perf = df["Performance_Score"].mean().round(2)
        st.metric("Avg Performance Score", f"{avg_perf}/5")
    with col3:
        promo_ready = (df["Promotion_Eligible"].sum() / len(df)) * 100
        st.metric("Promotion Readiness", f"{promo_ready:.1f}%")

    # --- Career Progression ---
    st.markdown("### Career Progression Insights")
    col4, col5 = st.columns(2)

    with col4:
        exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
        st.bar_chart(exp_by_level)

   with col5:
    # Pastikan kategori posisi dikonversi ke numerik untuk scatter
    df["Current_Position_Level"] = df["Current_Position_Level"].astype("category")
    level_codes = df["Current_Position_Level"].cat.codes
    level_labels = df["Current_Position_Level"].cat.categories

    fig, ax = plt.subplots()
    ax.scatter(
        level_codes,
        df["Performance_Score"],
        s=df["Promotion_Eligible"] * 200 + 40,
        alpha=0.6,
        c="#0078d4"
    )
    ax.set_xlabel("Current Position Level")
    ax.set_ylabel("Performance Score")
    ax.set_title("Performance vs. Promotion Eligibility")

    # Ganti angka di sumbu X dengan label posisi asli
    ax.set_xticks(range(len(level_labels)))
    ax.set_xticklabels(level_labels, rotation=30, ha="right")

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Predictive HR - Promotion")

    # ===============================
    # 6️⃣ TOP 5 PROMOTION RECOMMENDATION
    # ===============================
    df["Predicted_Promotion"] = model.predict(df.drop(columns=["Promotion_Eligible"], errors="ignore"))
    top5 = df[df["Predicted_Promotion"] == 1].head(5)
    st.markdown("### Top 5 Employees Recommended for Promotion")
    st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Predicted_Promotion"]])

    # ===============================
    # 7️⃣ CEK EMPLOYEE INDIVIDUAL
    # ===============================
    st.markdown("---")
    st.subheader("Check Employee Promotion Eligibility")

    emp_id_input = st.text_input("Enter Employee ID to check promotion status:")

    if emp_id_input:
        if emp_id_input in df["Employee_ID"].astype(str).values:
            st.success(f"Employee ID **{emp_id_input}** found in dataset.")

            emp_data = df[df["Employee_ID"].astype(str) == emp_id_input]
            st.dataframe(emp_data)

            prediction = model.predict(
                emp_data.drop(columns=["Promotion_Eligible", "Predicted_Promotion"], errors="ignore")
            )[0]

            if prediction == 1:
                st.markdown("### Eligible for Promotion")
            else:
                st.markdown("###Not Eligible for Promotion Yet")

            st.caption(f"Prediction generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        else:
            st.warning(f" Employee ID **{emp_id_input}** not found in dataset.")
            st.info("Please add this employee in the input form below or update the dataset.")

