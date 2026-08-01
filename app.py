import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Loads the pre-trained XGBoost model from the pickle file."""
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please place your pickle file in the root directory and rename it to 'model.pkl'.")
    st.stop()

st.title("🏦 Loan Approval Prediction System")
st.write("Fill in the applicant details below to predict loan approval eligibility.")

st.markdown("---")

# User Input Form
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Info")
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", options=["No", "Yes"])

    with col2:
        st.subheader("Loan & Credit Details")
        income_annum = st.number_input("Annual Income ($)", min_value=0, value=5000000, step=50000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000000, step=100000)
        loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=40, value=10, step=1)
        cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=750)

    with col3:
        st.subheader("Asset Values ($)")
        residential_assets_value = st.number_input("Residential Assets", min_value=0, value=4000000, step=50000)
        commercial_assets_value = st.number_input("Commercial Assets", min_value=0, value=2000000, step=50000)
        luxury_assets_value = st.number_input("Luxury Assets", min_value=0, value=10000000, step=50000)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=3000000, step=50000)

    submit_button = st.form_submit_button("Predict Loan Status", use_container_width=True)

if submit_button:
    # Encoding categorical values
    edu_encoded = 1 if education == "Graduate" else 0
    emp_encoded = 1 if self_employed == "Yes" else 0

    # Feature array construction matching the expected input order
    input_data = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'education': edu_encoded,
        'self_employed': emp_encoded,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }])

    # Prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Loan Approved!** (Approval Probability: {prediction_proba[1]:.2%})")
    else:
        st.error(f"❌ **Loan Rejected.** (Approval Probability: {prediction_proba[1]:.2%})")
