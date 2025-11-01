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

    # --- Metrics and Mini Pie ---
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    with col1:
        st.metric("Total Employees", len(df))
    with col2:
        avg_perf = df["Performance_Score"].mean().round(2)
        st.metric("Avg Performance Score", f"{avg_perf}/5")
    with col3:
        promo_counts = df["Promotion_Eligible"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(2.5, 2.5))
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

    st.markdown("---")
    st.markdown("### Career Progression Insights")

    # ===============================
    # Bar Chart 1: Performance Score
    # ===============================
    col4, col5 = st.columns(2)
    with col4:
        perf_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack(fill_value=0)
        fig_perf, ax_perf = plt.subplots()
        perf_data.plot(kind="bar", ax=ax_perf, color=["#ff7f0e", "#1f77b4"], width=0.7)
        ax_perf.set_xlabel("Current Position Level")
        ax_perf.set_ylabel("Avg Performance Score")
        ax_perf.set_title("Performance Score by Position & Eligibility")

        for container in ax_perf.containers:
            ax_perf.bar_label(container, fmt="%.2f", label_type="center")

        st.pyplot(fig_perf)

    # ===============================
    # Bar Chart 2: Leadership Score
    # ===============================
    with col5:
        if "Leadership_Score" in df.columns:
            lead_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Leadership_Score"].mean().unstack(fill_value=0)
            fig_lead, ax_lead = plt.subplots()
            lead_data.plot(kind="bar", ax=ax_lead, color=["#2ca02c", "#9467bd"], width=0.7)
            ax_lead.set_xlabel("Current Position Level")
            ax_lead.set_ylabel("Avg Leadership Score")
            ax_lead.set_title("Leadership Score by Position & Eligibility")

            for container in ax_lead.containers:
                ax_lead.bar_label(container, fmt="%.2f", label_type="center")

            st.pyplot(fig_lead)
        else:
            st.warning("⚠️ Column 'Leadership_Score' not found in dataset.")

    # ===============================
    # Boxplot: Project Handled
    # ===============================
    st.markdown("### Project Handled Distribution by Seniority Level")

    if "Project_Handled" in df.columns:
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        levels = df["Current_Position_Level"].unique()
        colors = {0: "#ff7f0e", 1: "#1f77b4"}

        for i, level in enumerate(levels):
            for elig in [0, 1]:
                data = df[(df["Current_Position_Level"] == level) & (df["Promotion_Eligible"] == elig)]["Project_Handled"]
                positions = [i * 2 + elig + 1]
                ax_box.boxplot(data, positions=positions, patch_artist=True,
                               boxprops=dict(facecolor=colors[elig], alpha=0.6),
                               medianprops=dict(color="black"))

        ax_box.set_xticks(range(1.5, len(levels)*2, 2))
        ax_box.set_xticklabels(levels, rotation=45, ha="right")
        ax_box.set_ylabel("Projects Handled")
        ax_box.set_title("Project Handled Distribution by Seniority Level")

        # Legend
        handles = [
            plt.Line2D([0], [0], color=colors[0], lw=4, label='Not Eligible'),
            plt.Line2D([0], [0], color=colors[1], lw=4, label='Eligible')
        ]
        ax_box.legend(handles=handles, title="Promotion Eligibility")
        st.pyplot(fig_box)
    else:
        st.warning("⚠️ Column 'Project_Handled' not found in dataset.")

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
    # ADMIN INPUT (RIGHT COLUMN)
    # ===============================
    st.mark
