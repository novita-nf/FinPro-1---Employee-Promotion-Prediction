# ======================================================
# app.py — HR Promotion Dashboard (Updated & Stabilized)
# ======================================================
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ===============================
# 1️⃣ CONFIG & THEME
# ===============================
st.set_page_config(page_title="HR Dashboard - ABC Company", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f7faff; }
[data-testid="stSidebar"] { background-color: #e8f1fa; color: black; }
[data-testid="stSidebar"] * { color: #0f4c81 !important; font-size: 15px !important; }
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
model_path = BASE_DIR / "model3.pkl"

# Load dataset
df = pd.read_csv(data_path, sep=";")

if "Projects_Handled" in df.columns:
    df = df.dropna(subset=["Projects_Handled"])

# Load trained pipeline
model = joblib.load(model_path)

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
menu = st.sidebar.radio(
    "Navigation",
    ["General Dashboard", "Talent Development", "Promotion", "Absency", "Recruitment"],
    index=2
)

# ===============================
# 5️⃣ GENERAL DASHBOARD
# ===============================
if menu == "General Dashboard":
    st.info("Welcome to the HR Dashboard. Please select a section from the sidebar.")

# ===============================
# 6️⃣ PROMOTION DASHBOARD
# ===============================
if menu == "Promotion":
    st.subheader("HR - Promotion Dashboard")

    # ========== METRIC CARDS ==========
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👥 Total Employees", len(df))

    with col2:
        avg_perf = df["Performance_Score"].mean().round(2)
        st.metric("⭐ Avg Performance Score", f"{avg_perf:.2f}/5")

        gauge = go.Figure(go.Indicator(mode="gauge+number",value=avg_perf,gauge={
          "axis": {"range": [0, 5]},
          "bar": {"color": "blue"}}, number={"suffix": " / 5"},title={"text": "Average Performance"}))

        gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(gauge, use_container_width=True)

    with col3:
        st.markdown("⭐Promotion Eligibility Distribution")
        
        promo_counts = df["Promotion_Eligible"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(2.2, 2.2))
        ax_pie.pie(
            promo_counts,
            labels=["Not Eligible", "Eligible"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#ff7f0e", "#1f77b4"]
        )
        st.pyplot(fig_pie)

    st.markdown("---")
    st.markdown("### Career Progression Insights")

    # ========== Tenure by Position ==========
    col4, col5, col6 = st.columns(3)

    with col4:
        exp_by_level = df.groupby("Current_Position_Level")["Years_at_Company"].mean()
        fig1, ax1 = plt.subplots()
        ax1.bar(exp_by_level.index, exp_by_level.values, color="#0078d4")
        ax1.set_xlabel("Current Position Level")
        ax1.set_ylabel("Average Years at Company")
        ax1.set_title("Average Tenure by Position Level")
        st.pyplot(fig1)

    # ========== Performance by Position & Eligibility ==========
    with col5:
        perf_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Performance_Score"].mean().unstack(fill_value=0)
        fig_perf, ax_perf = plt.subplots()
        perf_data.plot(kind="bar", ax=ax_perf, color=["#ff7f0e", "#1f77b4"], width=0.7)
        ax_perf.set_xlabel("Current Position Level")
        ax_perf.set_ylabel("Avg Performance Score")
        ax_perf.set_title("Performance Score by Position Level and Eligibility")
        for c in ax_perf.containers:
            ax_perf.bar_label(c, fmt="%.2f", label_type="center")
        st.pyplot(fig_perf)

    # ========== Leadership Score ==========
    with col6:
        lead_data = df.groupby(["Current_Position_Level", "Promotion_Eligible"])["Leadership_Score"].mean().unstack(fill_value=0)
        fig_lead, ax_lead = plt.subplots()
        lead_data.plot(kind="bar", ax=ax_lead, color=["#2ca02c", "#9467bd"], width=0.7)
        ax_lead.set_xlabel("Current Position Level")
        ax_lead.set_ylabel("Avg Leadership Score")
        ax_lead.set_title("Leadership Score by Position Level and Eligibility")
        for c in ax_lead.containers:
            ax_lead.bar_label(c, fmt="%.2f", label_type="center")
        st.pyplot(fig_lead)

    # ========== Projects Handled ==========
    st.markdown("### Projects Handled Distribution by Position and Eligibility")

    if "Projects_Handled" in df.columns:
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        levels = df["Current_Position_Level"].unique()
        colors = {0: "#ff7f0e", 1: "#1f77b4"}

        for i, level in enumerate(levels):
            for elig in [0, 1]:
                data = df[(df["Current_Position_Level"] == level) & (df["Promotion_Eligible"] == elig)]["Projects_Handled"]
                positions = [i * 2 + elig + 1]
                ax_box.boxplot(
                    data,
                    positions=positions,
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[elig], alpha=0.6),
                    medianprops=dict(color="black")
                )

        ax_box.set_xticks(np.arange(1.5, len(levels) * 2, 2))
        ax_box.set_xticklabels(levels, ha="right")
        ax_box.set_ylabel("Projects Handled")
        ax_box.set_title("Project Handled Distribution by Seniority Level")

        handles = [
            plt.Line2D([0], [0], color=colors[0], lw=4, label='Not Eligible'),
            plt.Line2D([0], [0], color=colors[1], lw=4, label='Eligible')
        ]
        ax_box.legend(handles=handles, title="Promotion Eligibility")
        st.pyplot(fig_box)

    st.markdown("---")

    # ===============================
    # 📌 MODEL INTERPRETATION SECTION
    # ===============================
    st.markdown("---")
    st.subheader("📘 Model Prediction Interpretation Guide")

    # Final interpretation
    st.markdown("""
    ### **🔍 Final Interpretation Summary**
    This section explains how model prediction outcomes should be interpreted and which employee groups require HR prioritization.  
    The model serves as a **decision-support tool**, not an automatic decision engine.

    It helps HR identify:
    - Employees with strong promotion-readiness signals  
    - Employees with missed potential near the threshold  
    - Employees who may be overestimated  
    - Employees not yet ready for promotion""")

    # Additional notes
    st.markdown("""
    ### **📝 Additional Notes**
    - **TP (> 0.70)** → High confidence. Strong promotion candidates  
    - **FN (0.40–0.50)** → High priority for manual review (possible missed talent)  
    - **FP (0.50–0.65)** → Requires data validation (possible overestimation)  
    - **TN (< 0.40)** → Low readiness, no urgent action required  
    These ranges help HR quickly classify employees using probability scores.""")

    # Interpretation Table
    st.markdown("### 📊 Interpretation Table")

    interp_df = pd.DataFrame({"Type": ["TP", "FN", "FP", "TN"],
                              "Percentage": ["24.40%", "4.80%", "4.80%", "66.00%"],
                              "Est. Count (per 500)": [122, 24, 24, 330],
                              "Probability Range": ["> 0.70", "0.40 – 0.50", "0.50 – 0.65", "< 0.40"],
                              "Category": ["Confirmed Eligible", "Missed Potential", "Overestimated", "Not Ready"],
                              "Interpretation": [
                                "Strong confidence — high likelihood of promotion readiness",
                                "Model missed some actual high performers near the threshold",
                                "Predicted eligible, but actual performance does not support it",
                                "Consistently low probability — unlikely to be promotion-ready"],
                              "HR Action": [
                                "Proceed to promotion review / fast-track",
                                "Manual review — may be overlooked strong performers",
                                "Validate performance records / check missing data",
                                "No immediate action needed"]
                             })
    st.dataframe(interp_df, use_container_width=True)

    # ========== Predictive HR ==========
    st.subheader("Predictive HR - Promotion")

    # Remove non-feature columns
    X = df.drop(columns=["Promotion_Eligible", "Employee_ID"], errors="ignore")

    # —— Add probability + label
    df["Predicted_Prob"] = model.predict_proba(X)[:, 1]
    df["Predicted_Label"] = model.predict(X)

    # ====== HR Interpretation Logic ======
    def classify_prediction(row):
      prob = row["Predicted_Prob"]
      actual = row["Promotion_Eligible"]
      pred = row["Predicted_Label"]

      # Prediction type
      if actual == 1 and pred == 1:
        pred_type = "TP"
      elif actual == 1 and pred == 0:
        pred_type = "FN"
      elif actual == 0 and pred == 1:
        pred_type = "FP"
      else:
        pred_type = "TN"

      # Probability categories
      if prob > 0.70:
        category = "Confirmed Eligible"
        action = "Proceed to promotion review / fast-track"
      elif 0.50 <= prob <= 0.65:
        category = "Overestimated"
        action = "Validate data completeness / performance records"
      elif 0.40 <= prob < 0.50:
        category = "Missed Potential"
        action = "Manual review — may be strong performers"
      else:
        category = "Not Ready"
        action = "No immediate action needed"

      return pd.Series([pred_type, category, action])

    # IMPORTANT: create new columns BEFORE top5
    df[["Prediction_Type", "Category", "HR_Action"]] = df.apply(classify_prediction, axis=1)

    # ===== SHOW TOP 5 =====
    st.markdown("### Top 5 Employees Recommended for Promotion (by probability)")

    top5 = df.sort_values("Predicted_Prob", ascending=False).head(5)

    st.table(top5[[
      "Employee_ID",
      "Current_Position_Level",
      "Performance_Score",
      "Predicted_Prob",
      "Category",
      "HR_Action"]])

    # ===== Individual Prediction =====
    st.markdown("---")
    st.subheader("Individual Employee Promotion Prediction")

    emp_id = st.text_input("Enter Employee ID (Format: EMPXXXX):")

    if emp_id:
      emp_row = df[df["Employee_ID"] == emp_id]
      if emp_row.empty:
        st.warning("Employee ID not found.")
      else:
        emp_X = emp_row.drop(
          columns=[
            "Promotion_Eligible",
            "Employee_ID",
            "Predicted_Prob",
            "Predicted_Label",
            "Prediction_Type",
            "Category",
            "HR_Action"],errors="ignore")
        prob = model.predict_proba(emp_X)[0][1]
        prediction = model.predict(emp_X)[0]

        st.success(f"""
        **Promotion Probability: {prob:.2f}**  
        **Category:** {emp_row['Category'].values[0]}  
        **Recommended HR Action:** {emp_row['HR_Action'].values[0]}
        """)

        
# ===============================
# 🔐 ADMIN SIDEBAR MENU
# ===============================
with st.sidebar:
    st.markdown("---")
    st.subheader("🔐 Data Input for Employee Promotion Prediction")
    password = st.text_input("Enter Admin Password", type="password")

    if password == "0000":
        st.success("Access Granted ✅")

        # Manual form
        emp_id_manual = st.text_input("Employee ID (Format: EMPXXXX)")
        position = st.selectbox("Position Level", df["Current_Position_Level"].unique())
        perf_score = st.number_input("Performance Score (1-5)", min_value=1, max_value=5)
        years = st.number_input("Years at Company", min_value=0, max_value=40)
        train_hours = st.number_input("Training Hours", min_value=0, max_value=500)
        leader_score = st.slider("Leadership Score", 1, 5)
        projects = st.number_input("Projects Handled", min_value=0, max_value=50)
        peer_score = st.slider("Peer Review Score", 1, 5)

        if st.button("Predict Manually"):
            new_row = pd.DataFrame([{
                "Current_Position_Level": position,
                "Performance_Score": perf_score,
                "Years_at_Company": years,
                "Training_Hours": train_hours,
                "Leadership_Score": leader_score,
                "Projects_Handled": projects,
                "Peer_Review_Score": peer_score,
                "Age": df["Age"].median()  # fallback if Age not provided
            }])

            prob = model.predict_proba(new_row)[0][1]
            pred = model.predict(new_row)[0]

            st.success(
                f"{emp_id_manual}: {'Eligible' if pred==1 else 'Not Eligible'} "
                f"(Prob: {prob:.2f})"
            )

        # —— Batch upload
        st.markdown("### 📤 Batch Upload CSV")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file, sep=";")
            batch_X = batch_df.drop(columns=["Employee_ID"], errors="ignore")
            batch_df["Predicted_Prob"] = model.predict_proba(batch_X)[:, 1]
            batch_df["Predicted_Promotion"] = model.predict(batch_X)
            st.success("Batch prediction completed.")
            st.dataframe(batch_df)

    else:
        st.info("Enter admin password to unlock admin tools.")
