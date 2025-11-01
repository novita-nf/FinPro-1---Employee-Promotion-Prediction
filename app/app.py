# ======================================================
# app.py — HR Promotion Dashboard (Final Full Version)
# ======================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import cloudpickle
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ CONFIG & THEME
# ===============================
st.set_page_config(page_title="HR Dashboard - ABC Company", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f7faff; }
[data-testid="stSidebar"] { background-color: #e8f1fa; color: black; }
[data-testid="stSidebar"] * { color: #0f4c81 !important; }
.main-header { background-color: #0078d4; padding: 12px; border-radius: 10px;
               color: white; font-size: 26px; font-weight: bold; text-align: center; }
h3 { color: #0f4c81; }
div[data-testid="stMetricValue"] { color: #0078d4; font-weight: 700; }
div.stButton > button { background-color: #0078d4; color: white; border-radius: 8px;
                        height: 3em; font-weight: 600; border: none; width: 100%; }
div.stButton > button:hover { background-color: #0f4c81; color: #f0f0f0; }
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
model_path = BASE_DIR / "model2.pkl"

df = pd.read_csv(data_path, sep=";")

with open(model_path, "rb") as f:
    model = cloudpickle.load(f)
  
# Drop rows with NaN in Project_Handled (for clean boxplot)
if "Project_Handled" in df.columns:
    df = df.dropna(subset=["Project_Handled"])
# ===============================
# 3️⃣ HEADER
# ===============================
logo_path = BASE_DIR / "ALGORANGER 2 Logo with Graph and Hat (1).png"
col1, col2 = st.columns([1, 5])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=85)
with col2:
    st.markdown('<div class="main-header">HR Dashboard - ABC Company</div>', unsafe_allow_html=True)

st.markdown("---")

# ===============================
# 4️⃣ SIDEBAR NAVIGATION
# ===============================
menu = st.sidebar.radio("Navigation",
    ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"],
    index=2)

# ===============================
# 5️⃣ PROMOTION DASHBOARD
# ===============================
if menu == "Promotion":
    st.subheader("HR - Promotion Dashboard")

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        avg_perf = df["Performance_Score"].mean().round(2)
        st.metric("Avg Performance Score", f"{avg_perf}/5")
    with col3:
        st.metric("Promotion Readiness", f"{promo_ready:.1f}%")
        st.markdown("### Promotion Eligibility Distribution") # --- Pie Chart: Promotion Eligible vs Not ---
        promo_counts = df["Promotion_Eligible"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(2.2, 2.2))
        ax_pie.pie(promo_counts, labels=["Not Eligible", "Eligible"], autopct="%1.1f%%", startangle=90, colors=["#ff7f0e", "#1f77b4"])
        st.pyplot(fig_pie)

    st.markdown("---")
    st.markdown("### Career Progression Insights")

    # ===============================
    # Bar Chart 1: Tenure per Position
    # ===============================
    col4, col5, col6 = st.columns(3)

    with col4:
        exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
        fig1, ax1 = plt.subplots()
        ax1.bar(exp_by_level.index, exp_by_level.values, color="#0078d4")
        ax1.set_xlabel("Current Position Level")
        ax1.set_ylabel("Average Years at Company")
        ax1.set_title("Average Tenure by Position Level")
        st.pyplot(fig1)
    # ===============================
    # Bar Chart 2: Performance Score
    # ===============================   
    with col5:
        perf_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack(fill_value=0)
        fig_perf, ax_perf = plt.subplots()
        perf_data.plot(kind="bar", ax=ax_perf, color=["#ff7f0e", "#1f77b4"], width=0.7)
        ax_perf.set_xlabel("Current Position Level")
        ax_perf.set_ylabel("Avg Performance Score")
        ax_perf.set_title("Performance Score by Position Level and Eligibility")

        for container in ax_perf.containers:
            ax_perf.bar_label(container, fmt="%.2f", label_type="center")

        st.pyplot(fig_perf)

    # ===============================
    # Bar Chart 3: Leadership Score
    # ===============================
    with col6:
        lead_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Leadership_Score"].mean().unstack(fill_value=0)
        fig_lead, ax_lead = plt.subplots()
        lead_data.plot(kind="bar", ax=ax_lead, color=["#2ca02c", "#9467bd"], width=0.7)
        ax_lead.set_xlabel("Current Position Level")
        ax_lead.set_ylabel("Avg Leadership Score")
        ax_lead.set_title("Leadership Score by Position Level and Eligibility")

        for container in ax_lead.containers:
            ax_lead.bar_label(container, fmt="%.2f", label_type="center")

        st.pyplot(fig_lead)

    # ===============================
    # Boxplot: Projects Handled
    # ===============================
    st.markdown("### Projects Handled Distribution by Position and Eligibility")
    fig_box, ax_box = plt.subplots(figsize=(8, 5))
    df["Group"] = df["Current_Position_Level"] + " - " + df["Promotion_Eligible"].map({0: "Not Eligible", 1: "Eligible"})
    df_sorted = df.sort_values("Current_Position_Level")
    box_data = [df_sorted[df_sorted["Group"] == g]["Projects_Handled"] for g in df_sorted["Group"].unique()]
    ax_box.boxplot(box_data, patch_artist=True,
                   boxprops=dict(facecolor="#1f77b4", alpha=0.5),
                   medianprops=dict(color="black"))
    ax_box.set_xticklabels(df_sorted["Group"].unique(), rotation=45, ha="right")
    ax_box.set_ylabel("Projects Handled")
    ax_box.set_title("Projects Handled by Position and Eligibility")
    st.pyplot(fig_box)

    st.markdown("---")

    # ===============================
    # Predictive HR Section
    # ===============================
    st.subheader("Predictive HR - Promotion")
    X = df.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
    df["Predicted_Promotion"] = model.predict(X)

    top5 = df[df["Predicted_Promotion"] == 1].head(5)
    st.markdown("### Top 5 Employees Recommended for Promotion")
    st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Predicted_Promotion"]])

    # ===============================
    # Individual Employee Prediction
    # ===============================
    st.markdown("---")
    st.subheader("Individual Employee Promotion Prediction")
    emp_id = st.text_input("Enter Employee ID (Format: EMPXXXX):")

    if emp_id:
        emp_row = df[df["Employee_ID"] == emp_id]
        if emp_row.empty:
            st.warning("Employee ID not found. Please check the format (e.g., EMP0001).")
        else:
            emp_X = emp_row.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
            prediction = model.predict(emp_X)[0]
            result = "✅ Eligible for Promotion" if prediction == 1 else "Not Eligible for Promotion Yet"
            st.success(f"**Prediction for {emp_id}: {result}**")

# ===============================
# 🔐 ADMIN SIDEBAR (KANAN)
# ===============================
with st.sidebar:
    st.markdown("---")
    st.subheader("🔐 Data Input for Employee Promotion Prediction")
    password = st.text_input("Enter Admin Password", type="password")

    if password == "0000":
        st.success("Access Granted ✅")

        emp_id_manual = st.text_input("Employee ID (Format: EMPXXXX)")
        position = st.selectbox("Position Level", df["Current_Position_Level"].unique())
        perf_score = st.number_input("Performance Score (1-5)", min_value=1, max_value=5, step=1)
        years = st.number_input("Years at Company", min_value=0, max_value=50, step=1)

        if st.button("Predict Manually"):
            new_row = pd.DataFrame({
                "Employee_ID": [emp_id_manual],
                "Current_Position_Level": [position],
                "Performance_Score": [perf_score],
                "Years_at_Company": [years]
            })
            new_X = new_row.drop(columns=["Employee_ID"])
            pred = model.predict(new_X)[0]
            st.success(f"{emp_id_manual}: {'✅ Eligible' if pred==1 else 'Not Eligible'}")

        st.markdown("### 📤 Batch Upload CSV")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file, sep=";")
                batch_X = batch_df.drop(columns=["Employee_ID"], errors="ignore")
                batch_df["Predicted_Promotion"] = model.predict(batch_X)
                st.success("✅ Batch prediction done!")
                st.dataframe(batch_df)
            except Exception as e:
                st.error(f"Error processing file: {e}")
    else:
        st.info("Enter admin password to unlock input features.")
