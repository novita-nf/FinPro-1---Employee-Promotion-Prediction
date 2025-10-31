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
