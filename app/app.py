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
    with
