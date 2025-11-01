import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===============================
# 1️⃣ CONFIG & THEME
# ===============================
st.set_page_config(page_title="HR Dashboard - ABC Company", layout="wide")

st.markdown("""
    <style>
        /* --- Sidebar kiri (Navigation) --- */
        [data-testid="stSidebarNav"] {
            background-color: #E3F2FD !important;
        }
        [data-testid="stSidebarNav"] * {
            font-size: 14px !important;
        }
        /* --- Sidebar kanan --- */
        section[data-testid="stSidebar"] {
            background-color: #F7F9FB;
        }
        /* --- Judul utama --- */
        h1, h2, h3 {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# ===============================
# 2️⃣ LOAD DATA & MODEL
# ===============================
model_path = Path("model1.pkl")
if model_path.exists():
    model = joblib.load(model_path)
else:
    model = None

df_path = Path("data/employee_data.csv")
if df_path.exists():
    df = pd.read_csv(df_path)
else:
    st.warning("⚠️ Data not found. Please upload 'employee_data.csv' in the /data folder.")
    df = pd.DataFrame()

# Pastikan kolom minimal ada
required_cols = [
    "Age", "Years_at_Company", "Performance_Score", "Leadership_Score",
    "Training_Hours", "Projects_Handled", "Peer_Review_Score",
    "Current_Position_Level", "Promotion_Eligible"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.warning(f"⚠️ Missing columns in dataset: {missing_cols}")
else:
    # Hanya drop jika kolom ada
    df = df.dropna(subset=["Current_Position_Level", "Projects_Handled"], how="any")

# ===============================
# 3️⃣ SIDEBAR NAVIGATION (KIRI)
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["General Dashboard", "Career Progression Insight", "Promotion Eligibility"])

# ===============================
# 4️⃣ PAGE 1: GENERAL DASHBOARD
# ===============================
if page == "General Dashboard":
    st.title("📊 General Employee Dashboard")

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            avg_perf = df["Performance_Score"].mean()
            st.metric("Avg Performance Score", f"{avg_perf:.2f}")
        with col3:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            eligible_rate = df["Promotion_Eligible"].mean()
            ax.pie([eligible_rate, 1 - eligible_rate],
                labels=[f"Eligible ({eligible_rate*100:.1f}%)", "Not Eligible"],
                colors=["#4CAF50", "#FFC107"],
                autopct="%1.1f%%", startangle=90)
            st.pyplot(fig)
    else:
        st.info("Upload dataset to view dashboard.")

# ===============================
# 5️⃣ PAGE 2: CAREER PROGRESSION INSIGHT
# ===============================
elif page == "Career Progression Insight":
    st.title("📈 Career Progression Insights")

    if not df.empty:
        # Leadership Score by Position
        st.subheader("Leadership Score by Position and Promotion Eligibility")
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=df,
            x="Current_Position_Level",
            y="Leadership_Score",
            hue="Promotion_Eligible",
            palette=["#42A5F5", "#9CCC65"]
        )
        plt.xlabel("Position Level")
        plt.ylabel("Avg Leadership Score")
        plt.legend(title="Eligible")
        st.pyplot(plt)

        # Seniority vs Years
        st.subheader("Seniority Level vs Years at Company")
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Current_Position_Level",
            y="Years_at_Company",
            palette="Blues"
        )
        plt.xlabel("Seniority Level")
        plt.ylabel("Years at Company")
        st.pyplot(plt)
    else:
        st.info("Upload dataset to view insights.")

# ===============================
# 6️⃣ PAGE 3: PROMOTION ELIGIBILITY
# ===============================
elif page == "Promotion Eligibility":
    st.title("🎯 Promotion Eligibility Dashboard")

    # Sidebar kanan khusus halaman ini
    with st.sidebar:
        st.subheader("Data Input for Employee Promotion Prediction")
        password = st.text_input("Enter Admin Password", type="password")
        if password == "admin123":
            st.success("Access Granted ✅")
            st.text_input("Enter Employee ID (Format: EMPXXXX)")
            st.number_input("Age", min_value=18, max_value=65, step=1)
            st.number_input("Years at Company", min_value=0, step=1)
            st.number_input("Performance Score", min_value=0.0, max_value=5.0, step=0.1)
            st.number_input("Leadership Score", min_value=0.0, max_value=5.0, step=0.1)
            st.number_input("Training Hours", min_value=0, step=1)
            st.number_input("Projects Handled", min_value=0, step=1)
            st.number_input("Peer Review Score", min_value=0.0, max_value=5.0, step=0.1)
            st.selectbox("Current Position Level", df["Current_Position_Level"].unique() if "Current_Position_Level" in df.columns else [])
            st.button("Predict Promotion Eligibility")
        else:
            st.warning("🔒 Enter password to access admin input")

    # Projects Handled Distribution
    if not df.empty:
        st.subheader("Projects Handled Distribution by Seniority Level")
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x="Current_Position_Level",
            y="Projects_Handled",
            hue="Promotion_Eligible",
            palette=["#42A5F5", "#9CCC65"]
        )
        plt.xlabel("Seniority Level")
        plt.ylabel("Projects Handled")
        plt.legend(title="Eligible", loc="upper right")
        st.pyplot(plt)
    else:
        st.info("Upload dataset to view promotion insights.")
