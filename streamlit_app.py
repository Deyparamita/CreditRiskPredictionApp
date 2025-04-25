import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

st.title("💳 Credit Risk Prediction App")
st.write("Fill in the applicant's details to predict the credit risk.")

# Mapping for Job
job_mapping = {
    "Unskilled and non-resident": 0,
    "Unskilled and resident": 1,
    "Skilled employee": 2,
    "Highly skilled (management/self-employed)": 3
}


# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job_label = st.selectbox("Job Type", list(job_mapping.keys()))
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_acc = st.selectbox("Saving Accounts", ["no_info", "little", "moderate", "rich"])
checking_acc = st.selectbox("Checking Account", ["no_info", "little", "moderate", "rich"])
credit_amount = st.text_input("Credit Amount", "1000")
duration = st.text_input("Duration (months)", "12")
purpose = st.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances"])

# Create input data
try:
    credit_amount = int(credit_amount)  
    duration = int(duration)  
    input_data = {
        'Age': age,
        'Sex': label_encoders['Sex'].transform([sex])[0],
        'Job': job_mapping[job_label],
        'Housing': label_encoders['Housing'].transform([housing])[0],
        'Saving accounts': label_encoders['Saving accounts'].transform([saving_acc])[0],
        'Checking account': label_encoders['Checking account'].transform([checking_acc])[0],
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': label_encoders['Purpose'].transform([purpose])[0],
        'Credit_per_month': credit_amount / (duration + 1)  # same as during training
    }
except KeyError as e:
    st.error(f"❌ Encoder for column '{str(e)}' not found. Please check your preprocessing or saved encoders.")
    st.stop()

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict Credit Risk"):
    prediction = model.predict(scaled_input)[0]
    if prediction == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")
