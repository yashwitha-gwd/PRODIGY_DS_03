import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))
columns = pickle.load(open('../models/columns.pkl', 'rb'))

st.title("Customer Purchase Prediction")

st.write("Enter customer details:")

# Example inputs (keep simple for now)
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", 0, 100000, 1000)
campaign = st.slider("Campaign Contacts", 1, 50, 2)

# Create input dataframe (you may need to match your features)
input_data = pd.DataFrame({
    'age': [age],
    'balance': [balance],
    'campaign': [campaign]
})

# Predict
if st.button("Predict"):
    input_data = input_data.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(input_data)[:,1][0]
    
    if prob > 0.05:
        st.success(f"Likely to Purchase (Confidence: {prob:.2f})")
    else:
        st.error(f"Unlikely to Purchase (Confidence: {prob:.2f})")
