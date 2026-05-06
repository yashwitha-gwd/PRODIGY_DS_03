import streamlit as st
import pandas as pd

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import predict  # ✅ use your pipeline

st.set_page_config(page_title="Customer Purchase Prediction")

st.title("📊 Customer Purchase Prediction")

st.write("Enter customer details:")

# ----------------------------
# User Inputs (IMPORTANT: include key features)
# ----------------------------
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", 0, 100000, 1000)
campaign = st.slider("Campaign Contacts", 1, 50, 2)

job = st.selectbox("Job", ["admin.", "technician", "services", "management"])
marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary"])

# ----------------------------
# Create input DataFrame
# ----------------------------
input_data = pd.DataFrame({
    'age': [age],
    'balance': [balance],
    'campaign': [campaign],
    'job': [job],
    'marital': [marital],
    'education': [education]
})

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    result, prob = predict(input_data)

    if prob > 0.5:
        st.success(f"✅ {result} (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ {result} (Confidence: {prob:.2f})")
