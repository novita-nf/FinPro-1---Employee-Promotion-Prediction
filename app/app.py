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
# 2️⃣ LOAD DATA & MODEL
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
    try:
        df_model = df.select_dtypes(include=["number"]).copy()
        df["Predicted_Promotion"] = model.predict(df_model)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        df["Predicted_Promotion"] = 0

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

            # Pastikan kolom sesuai model
            emp_features = emp_data.select_dtypes(include=["number"])
            prediction = model.predict(emp_features)[0]

            if prediction == 1:
                st.markdown("### ✅ Eligible for Promotion")
            else:
                st.markdown("### ⚠️ Not Eligible for Promotion Yet")

            st.caption(f"Prediction generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

        else:
            st.warning(f"Employee ID **{emp_id_input}** not found in dataset.")
            st.info("Please add this employee in the input form below or upload CSV.")


    # ===============================
    # 8️⃣ INPUT SECTION (MANUAL + CSV)
    # ===============================
    st.markdown("---")
    st.subheader("🔒 Employee Input Prediction")

    login_user = st.text_input("Login ID")
    login_pass = st.text_input("Password", type="password")

    if login_user == "0000" and login_pass == "0000":
        st.success("Login successful ✅")

        st.markdown("### 📋 Add New Employee Data Manually or Upload CSV")

        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload CSV"]
        )

        if input_method == "Manual Entry":
            colA, colB = st.columns(2)
            with colA:
                emp_id = st.text_input("Employee ID")
                age = st.number_input("Age", 18, 65, 30)
                years = st.number_input("Years at Company", 0, 40, 5)
                perf = st.number_input("Performance Score (1-5)", 1, 5, 3)
                lead = st.number_input("Leadership Score (0-100)", 0, 100, 50)
            with colB:
                train = st.number_input("Training Hours", 0, 200, 40)
                projects = st.number_input("Projects Handled", 0, 50, 5)
                peer = st.number_input("Peer Review Score (0-100)", 0, 100, 60)
                pos = st.selectbox("Current Position Level", ["Junior", "Mid", "Senior", "Lead"])

            if st.button("Predict Promotion Eligibility"):
                new_data = pd.DataFrame([{
                    "Age": age,
                    "Years_at_Company": years,
                    "Performance_Score": perf,
                    "Leadership_Score": lead,
                    "Training_Hours": train,
                    "Projects_Handled": projects,
                    "Peer_Review_Score": peer,
                }])

                prediction = model.predict(new_data)[0]
                if prediction == 1:
                    st.success("✅ This employee is likely **Eligible for Promotion**.")
                else:
                    st.warning("This employee is **Not Yet Eligible** for promotion.")

        elif input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                new_df = pd.read_csv(uploaded_file, sep=";")
                st.write("Uploaded Data Preview:", new_df.head())

                try:
                    new_df_model = new_df.select_dtypes(include=["number"])
                    preds = model.predict(new_df_model)
                    new_df["Predicted_Promotion"] = preds
                    st.success("✅ Predictions generated successfully!")
                    st.dataframe(new_df)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    else:
        st.info("Please login with ID: 0000 and Password: 0000 to access prediction input.")
