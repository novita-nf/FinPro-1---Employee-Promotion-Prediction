# app.py
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ CONFIG & THEME
# ===============================
st.set_page_config(
    page_title="HR Dashboard - ABC Company",
    layout="wide"
)

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
# 2️⃣ LOAD DATA & MODEL
# ===============================
if "__file__" in globals():
    BASE_DIR = Path(__file__).parent
else:
    BASE_DIR = Path(os.getcwd())

data_path = BASE_DIR.parent / "Data" / "Rakamin Bootcamp - Dataset - Promotion Dataset.csv"
model_path = BASE_DIR / "model.pkl"

df = pd.read_csv(data_path, sep=";")
model = joblib.load(model_path)

# ===============================
# 3️⃣ HEADER
# ===============================
logo_path = Path(__file__).parent / "ALGORANGER 2 Logo with Graph and Hat (1).png"

col1, col2 = st.columns([1, 5])
with col1:
    st.image(str(logo_path), width=85)
with col2:
    st.markdown('<div class="main-header">HR Dashboard - ABC Company</div>', unsafe_allow_html=True)

st.markdown("---")

# ===============================
# 4️⃣ SIDEBAR NAVIGATION
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"],
    index=2
)

# ===============================
# 5️⃣ TAB: PROMOTION DASHBOARD
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
    # --- Career Progression ---
st.markdown("### Career Progression Insights")

col4, col5 = st.columns(2)

with col4:
    exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
    fig1, ax1 = plt.subplots()
    ax1.bar(exp_by_level.index, exp_by_level.values, color="#0078d4")
    ax1.set_xlabel("Current Position Level")
    ax1.set_ylabel("Average Years at Company")
    ax1.set_title("Average Tenure by Position Level")
    st.pyplot(fig1)

with col5:
    stacked_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack(fill_value=0)
    stacked_data.plot(kind="bar", stacked=True, color=["#ff7f0e", "#1f77b4"])
    plt.xlabel("Current Position Level")
    plt.ylabel("Average Performance Score")
    plt.title("Performance Score by Position Level and Promotion Eligibility")
    st.pyplot(plt.gcf())

st.markdown("---")
st.subheader("Predictive HR - Promotion")

# ====================================
# 6️⃣ DATA PREPARATION FOR PREDICTION
# ====================================
# Simulasi encoding seperti training (pastikan sama)
X = df.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
X = pd.get_dummies(X, drop_first=True)

# Cek kesesuaian kolom dengan model
if hasattr(model, "feature_names_in_"):
    missing_cols = set(model.feature_names_in_) - set(X.columns)
    for col in missing_cols:
        X[col] = 0  # tambahkan kolom kosong bila hilang
    X = X[model.feature_names_in_]

# ====================================
# 7️⃣ TOP 5 PROMOTION RECOMMENDATION
# ====================================
df["Predicted_Promotion"] = model.predict(X)
top5 = df[df["Predicted_Promotion"] == 1].head(5)
st.markdown("### Top 5 Employees Recommended for Promotion")
st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Predicted_Promotion"]])

# ====================================
# 8️⃣ PREDIKSI BERDASARKAN EMPLOYEE ID
# ====================================
st.markdown("---")
st.subheader("🔍 Individual Employee Promotion Prediction")
emp_id = st.text_input("Enter Employee ID (Format: EMPXXXX):")

if emp_id:
    emp_row = df[df["Employee_ID"] == emp_id]
    if emp_row.empty:
        st.warning("Employee ID not found. Please check the format (e.g., EMP0001).")
    else:
        emp_X = emp_row.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
        emp_X = pd.get_dummies(emp_X, drop_first=True)

        # Samakan kolomnya dengan model
        for col in missing_cols:
            if col not in emp_X.columns:
                emp_X[col] = 0
        emp_X = emp_X[model.feature_names_in_]

        prediction = model.predict(emp_X)[0]
        result = "✅ Eligible for Promotion" if prediction == 1 else "❌ Not Eligible for Promotion"
        st.success(f"**Prediction for {emp_id}: {result}**")


    # ===============================
    # 8️⃣ UPLOAD CSV FOR BATCH PREDICTION
    # ===============================
    st.markdown("---")
    st.subheader("📤 Upload CSV for Batch Promotion Prediction")

    uploaded_file = st.file_uploader("Upload your employee data (CSV with same structure)", type=["csv"])

    if uploaded_file:
        try:
            new_df = pd.read_csv(uploaded_file, sep=None, engine="python")
            st.write("✅ File uploaded successfully! Preview:")
            st.dataframe(new_df.head())

            # Align columns to model features
            new_X = new_df.reindex(columns=feature_cols, fill_value=0)
            new_df["Predicted_Promotion"] = model.predict(new_X)

            st.markdown("### Prediction Results")
            st.dataframe(new_df[["Employee_ID", "Predicted_Promotion"]])

            # Summary chart
            promo_summary = new_df["Predicted_Promotion"].value_counts()
            fig, ax = plt.subplots()
            promo_summary.plot(kind="bar", color=["#f28e2b", "#4e79a7"], ax=ax)
            ax.set_title("Promotion Prediction Summary")
            ax.set_xlabel("Predicted Promotion (0=No, 1=Yes)")
            ax.set_ylabel("Employee Count")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error processing uploaded file: {e}")
