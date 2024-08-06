import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model
log_model = joblib.load('log_model.joblib')
rf_model = joblib.load('rf_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')

# Streamlit app
st.title("Customer Churn Prediction")

# Model selection
model_choice = st.selectbox("Select Model", ['Logistic Regression', 'Random Forest', 'XGBoost'])

# Define the input fields with your own column names
account_length = st.number_input("Account Length", min_value=0, max_value=300, value=1)
area_code = st.number_input("Area Code", min_value=0, max_value=999, value=1)
international_plan = st.selectbox("International Plan", ['No', 'Yes'])
voice_mail_plan = st.selectbox("Voice Mail Plan", ['No', 'Yes'])
number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, value=1)
number_customer_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, max_value=20, value=1)
average_call_duration = st.number_input("Average Call Duration", min_value=0.0, max_value=100.0, value=1.0)
total_charge = st.number_input("Total Charge", min_value=0.0, max_value=10000.0, value=1.0)
total_calls = st.number_input("Total Calls", min_value=0, max_value=1000, value=1)
total_minutes = st.number_input("Total Minutes", min_value=0.0, max_value=10000.0, value=1.0)
service_usage = st.number_input("Service Usage", min_value=0.0, max_value=10000.0, value=1.0)

# Convert inputs to model input format
input_data = np.array([[account_length, area_code, international_plan == 'Yes', voice_mail_plan == 'Yes',
                        number_vmail_messages, number_customer_service_calls, average_call_duration, total_charge,
                        total_calls, total_minutes, service_usage]])

# Select the model based on user choice
if model_choice == 'Logistic Regression':
    model = log_model
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Prediction: {'Churn' if prediction[0] else 'No Churn'}")
        st.write(f"Probability of Churn: {probability:.2f}")

elif model_choice == 'Random Forest':
    
    model = rf_model
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Prediction: {'Churn' if prediction[0] else 'No Churn'}")
        st.write(f"Probability of Churn: {probability:.2f}")

        # Display feature importance
        feature_importance = pd.Series(model.feature_importances_, index=['account_length', 'area_code', 'international_plan',
                                                                      'voice_mail_plan', 'number_vmail_messages', 'number_customer_service_calls',
                                                                      'average_call_duration', 'total_charge', 'total_calls', 'total_minutes',
                                                                      'service_usage'])
        st.bar_chart(feature_importance.nlargest(10))
else:
    model = xgb_model
    if st.button("Predict Churn"):
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        st.write(f"Prediction: {'Churn' if prediction[0] else 'No Churn'}")
        st.write(f"Probability of Churn: {probability:.2f}")

        # Display feature importance
        feature_importance = pd.Series(model.feature_importances_, index=['account_length', 'area_code', 'international_plan',
                                                                      'voice_mail_plan', 'number_vmail_messages', 'number_customer_service_calls',
                                                                      'average_call_duration', 'total_charge', 'total_calls', 'total_minutes',
                                                                      'service_usage'])
        st.bar_chart(feature_importance.nlargest(10))
