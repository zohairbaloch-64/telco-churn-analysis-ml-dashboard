import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Churn Analytics SaaS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.metric-card {
    background-color: #111;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}
.metric-title {
    font-size: 14px;
    color: #aaa;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

# -------------------- CLEANING --------------------
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.title("🔍 Filters")

tenure_range = st.sidebar.slider(
    "Tenure",
    int(df["tenure"].min()),
    int(df["tenure"].max()),
    (1, 50)
)

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

filtered_df = df[
    (df["tenure"] >= tenure_range[0]) &
    (df["tenure"] <= tenure_range[1]) &
    (df["Contract"].isin(contract_filter))
]

# -------------------- HEADER --------------------
st.title("📊 Customer Churn Analytics Dashboard")
st.caption("Analyze customer behavior, identify churn risks, and predict customer retention.")

# -------------------- KPI SECTION --------------------
col1, col2, col3, col4 = st.columns(4)

total_customers = len(filtered_df)
churn_rate = (filtered_df["Churn"] == "Yes").mean() * 100
avg_charge = filtered_df["MonthlyCharges"].mean()
avg_tenure = filtered_df["tenure"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"{avg_charge:.2f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f}")

st.markdown("---")

# -------------------- INSIGHTS SECTION --------------------
st.subheader("📊 Key Insights")

st.info("Customers with month-to-month contracts show significantly higher churn.")
st.info("Higher monthly charges are associated with increased churn risk.")
st.info("New customers (low tenure) are more likely to churn.")

# -------------------- CHARTS --------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(filtered_df, x="Contract", color="Churn",
                        barmode="group", title="Churn by Contract Type")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.box(filtered_df, x="Churn", y="MonthlyCharges",
                  title="Monthly Charges vs Churn")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(filtered_df, x="tenure", color="Churn",
                        title="Tenure Distribution")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.pie(filtered_df, names="Churn", title="Churn Distribution")
    st.plotly_chart(fig4, use_container_width=True)

# -------------------- MODEL TRAINING --------------------
df_model = df.copy()
encoders = {}

categorical_cols = [col for col in df_model.columns if df_model[col].dtype == "object" and col != "Churn"]
for col in categorical_cols:
    encoder = LabelEncoder()
    df_model[col] = encoder.fit_transform(df_model[col])
    encoders[col] = encoder

# Encode target separately so we keep X feature names aligned
target_encoder = LabelEncoder()
df_model["Churn"] = target_encoder.fit_transform(df_model["Churn"])

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]
feature_columns = X.columns.tolist()
dtypes = X.dtypes

# Convert to numpy arrays to avoid pandas conversion issues in sklearn
X = X.values
y = y.values

# Ensure no NaN or inf values
if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
    st.error("Data contains NaN or infinite values after preprocessing. Please check the data.")
    st.stop()

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# -------------------- FEATURE IMPORTANCE --------------------
st.markdown("---")
st.subheader("📈 Feature Importance")

importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(feat_df.head(10),
                 x="Importance", y="Feature",
                 orientation="h",
                 title="Top Features Influencing Churn")

st.plotly_chart(fig_imp, use_container_width=True)

# -------------------- PREDICTION SECTION --------------------
st.markdown("---")
st.subheader("🤖 Churn Prediction")

# User-friendly inputs
gender = st.selectbox("Gender", ["Female", "Male"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tenure = st.slider("Tenure", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=1000.0)

# Create input dict (minimal for demo)
input_dict = {
    "gender": gender,
    "Contract": contract,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Fill missing columns with default values
for col in feature_columns:
    if col not in input_dict:
        input_dict[col] = df[col].mode()[0]

input_df = pd.DataFrame([input_dict])

# Encode input same as training
for col in input_df.columns:
    if input_df[col].dtype == "object" and col in encoders:
        encoder = encoders[col]
        try:
            input_df[col] = encoder.transform(input_df[col])
        except ValueError:
            # Unknown category fallback: use the most common training label
            default_label = encoder.classes_[0]
            input_df[col] = input_df[col].apply(lambda x: x if x in encoder.classes_ else default_label)
            input_df[col] = encoder.transform(input_df[col])

# Align features exactly with the model training order
default_values = {col: df[col].mode().iloc[0] for col in feature_columns}
input_df = input_df.reindex(columns=feature_columns)
input_df = input_df.fillna(default_values)
input_df = input_df.astype(dtypes.to_dict())

# Prediction
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Churn Probability", f"{prob*100:.2f}%")

    if pred == 1:
        st.error("⚠️ High Risk: Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

# -------------------- DOWNLOAD --------------------
st.markdown("---")
st.download_button("📥 Download Filtered Data",
                   filtered_df.to_csv(index=False),
                   file_name="filtered_data.csv")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with Streamlit | Data Analyst Portfolio Project")