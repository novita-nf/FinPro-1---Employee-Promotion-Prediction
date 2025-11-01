import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
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

# Drop NaN untuk kolom Projects_Handled (agar boxplot tidak error)
if "Projects_Handled" in df.columns:
    df = df.dropna(subset=["Projects_Handled"])

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
# 4️⃣ LAYOUT: LEFT NAV + MAIN + RIGHT ADMIN
# ===============================
left_col, main_col, right_col = st.columns([1.2, 3.6, 1.2])

with left_col:
    st.markdown("### 🧭 Navigation")
    menu = st.radio("Go to", ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"], index=2)

# ===============================
# 5️⃣ MAIN DASHBOARD
# ===============================
with main_col:
    if menu == "Promotion":
        st.subheader("HR - Promotion Dashboard")

        # --- Metrics + Mini Pie ---
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            avg_perf = df["Performance_Score"].mean().round(2)
            st.metric("Avg Performance Score", f"{avg_perf}/5")
        with col3:
            promo_counts = df["Promotion_Eligible"].value_counts()
            fig_pie, ax_pie = plt.subplots(figsize=(2.4, 2.4))
            ax_pie.pie(
                promo_counts,
                labels=["Not Eligible", "Eligible"],
                autopct="%1.1f%%",
                startangle=90,
                textprops={'fontsize': 9},
                colors=["#ff7f0e", "#1f77b4"]
            )
            ax_pie.set_title("Promotion Eligibility", fontsize=10)
            st.pyplot(fig_pie)

        # ===============================
        # CAREER PROGRESSION INSIGHTS
        # ===============================
        st.markdown("---")
        st.markdown("### Career Progression Insights")

        colA, colB, colC = st.columns(3)

        # --- Average Tenure (Years_at_Company) ---
        with colA:
            exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
            fig1, ax1 = plt.subplots()
            ax1.bar(exp_by_level.index, exp_by_level.values, color="#0078d4")
            ax1.set_xlabel("Current Position Level")
            ax1.set_ylabel("Avg Years at Company")
            ax1.set_title("Average Tenure by Position Level")
            for i, val in enumerate(exp_by_level.values):
                ax1.text(i, val + 0.2, f"{val:.1f}", ha="center")
            st.pyplot(fig1)

        # --- Performance Score by Level & Eligibility ---
        with colB:
            perf_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack(fill_value=0)
            fig_perf, ax_perf = plt.subplots()
            perf_data.plot(kind="bar", ax=ax_perf, color=["#ff7f0e", "#1f77b4"], width=0.7)
            ax_perf.set_xlabel("Position Level")
            ax_perf.set_ylabel("Avg Performance Score")
            ax_perf.set_title("Performance Score by Position & Eligibility")
            for container in ax_perf.containers:
                ax_perf.bar_label(container, fmt="%.2f", label_type="center")
            st.pyplot(fig_perf)

        # --- Leadership Score by Level & Eligibility ---
        with colC:
            if "Leadership_Score" in df.columns:
                lead_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Leadership_Score"].mean().unstack(fill_value=0)
                fig_lead, ax_lead = plt.subplots()
                lead_data.plot(kind="bar", ax=ax_lead, color=["#2ca02c", "#9467bd"], width=0.7)
                ax_lead.set_xlabel("Position Level")
                ax_lead.set_ylabel("Avg Leadership Score")
                ax_lead.set_title("Leadership Score by Position & Eligibility")
                for container in ax_lead.containers:
                    ax_lead.bar_label(container, fmt="%.2f", label_type="center")
                st.pyplot(fig_lead)
            else:
                st.warning("⚠️ Column 'Leadership_Score' not found in dataset.")

        # ===============================
        # PROJECTS HANDLED BOX PLOT
        # ===============================
        st.markdown("---")
        st.markdown("### Projects Handled Distribution by Seniority Level")

        if "Projects_Handled" in df.columns:
            levels = sorted(df["Current_Position_Level"].unique())
            colors = {0: "#ff7f0e", 1: "#1f77b4"}
            fig_box, ax_box = plt.subplots(figsize=(8, 5))

            for i, level in enumerate(levels):
                for elig in [0, 1]:
                    subset = df[(df["Current_Position_Level"] == level) & (df["Promotion_Eligible"] == elig)]
                    data = subset["Projects_Handled"]
                    if not data.empty:
                        pos = i * 2 + elig + 1
                        ax_box.boxplot(data, positions=[pos], patch_artist=True,
                                       boxprops=dict(facecolor=colors[elig], alpha=0.6),
                                       medianprops=dict(color="black"))

            ax_box.set_xticks([i * 2 + 1.5 for i in range(len(levels))])
            ax_box.set_xticklabels(levels, rotation=45, ha="right")
            ax_box.set_ylabel("Projects Handled")
            ax_box.set_title("Projects Handled by Seniority Level")
            handles = [
                plt.Line2D([0], [0], color=colors[0], lw=4, label='Not Eligible'),
                plt.Line2D([0], [0], color=colors[1], lw=4, label='Eligible')
            ]
            ax_box.legend(handles=handles, title="Promotion Eligibility")
            st.pyplot(fig_box)

        # ===============================
        # PREDICTIVE HR - TOP 5
        # ===============================
        st.markdown("---")
        st.subheader("Predictive HR - Promotion")
        X = df.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")
        df["Predicted_Promotion"] = model.predict(X)
        top5 = df[df["Predicted_Promotion"] == 1].head(5)
        st.markdown("### Top 5 Employees Recommended for Promotion")
        st.table(top5[["Employee_ID", "Current_Position_Level", "Performance_Score", "Predicted_Promotion"]])

        # ===============================
        # INDIVIDUAL EMPLOYEE PREDICTION
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
                try:
                    prediction = model.predict(emp_X)[0]
                    result = "✅ Eligible for Promotion" if prediction == 1 else "Not Eligible for Promotion Yet"
                    st.success(f"Prediction for {emp_id}: {result}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ===============================
# 🔐 RIGHT SIDEBAR (ADMIN PANEL)
# ===============================
with right_col:
    st.markdown("### 🔐 Data Input for Employee Promotion Prediction")
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
