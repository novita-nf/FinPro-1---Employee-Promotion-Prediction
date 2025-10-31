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
    st.markdown("### Career Progression Insights")
    col4, col5 = st.columns(2)

    with col4:
        exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
        fig, ax = plt.subplots()
        exp_by_level.plot(kind="bar", color="#0078d4", ax=ax)
        ax.set_xlabel("Current Position Level")
        ax.set_ylabel("Average Years at Company")
        ax.set_title("Average Experience by Position Level")
        st.pyplot(fig)

    with col5:
        # Stacked bar: Performance Score vs Promotion Eligibility
        perf_stack = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack()
        perf_stack.plot(
            kind="bar",
            stacked=True,
            color=["#f28e2b", "#4e79a7"],
            figsize=(6, 4)
        )
        plt.title("Performance Score by Position Level (Stacked by Eligibility)")
        plt.xlabel("Current Position Level")
        plt.ylabel("Average Performance Score")
        plt.legend(["Not Eligible", "Eligible"])
        st.pyplot(plt)

    st.markdown("---")
    st.subheader("Predictive HR - Promotion")

    # --- Align features for prediction ---
    feature_cols = (
        model.feature_names_in_
        if hasattr(model, "feature_names_in_")
        else df.drop(columns=["Promotion_Eligible"], errors="ignore").columns
    )
    X = df.reindex(columns=feature_cols, fill_value=0)

    df["Predicted_Promotion"] = model.predict(X)
    top5 = df[df["Predicted_Promotion"] == 1].head(5)

    st.markdown("### Top 5 Employees Recommended for Promotion")
    st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Predicted_Promotion"]])

    # ===============================
    # 7️⃣ CEK EMPLOYEE INDIVIDUAL
    # ===============================
    st.markdown("---")
    st.subheader("Check Employee Promotion Eligibility (Format: EMPXXXX)")

    emp_id_input = st.text_input("Enter Employee ID:")

    if emp_id_input:
        if emp_id_input in df["Employee_ID"].astype(str).values:
            emp_data = df[df["Employee_ID"].astype(str) == emp_id_input]
            st.success(f"Employee ID **{emp_id_input}** found in dataset.")
            st.dataframe(emp_data)

            emp_features = emp_data.reindex(columns=feature_cols, fill_value=0)

            try:
                prediction = model.predict(emp_features)[0]
                if prediction == 1:
                    st.markdown("### ✅ Eligible for Promotion")
                else:
                    st.markdown("### ❌ Not Eligible for Promotion Yet")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

            st.caption(f"Prediction generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning(f"Employee ID **{emp_id_input}** not found in dataset.")
            st.info("Please add this employee in the input form below or upload CSV below.")

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
