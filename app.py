#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("heart_disease_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Function to preprocess and make predictions
def preprocess_and_predict(features):
    # Convert input features to a DataFrame to ensure column compatibility with the trained model
    input_data = pd.DataFrame([features])

    # One-Hot Encoding for 'sex' column (creates 'sex_male' as 1 for male, 0 for female)
    input_data = pd.get_dummies(input_data, columns=['sex'], drop_first=True)
    
    # Ensure compatibility with model's feature columns by adding missing columns as 0
    required_columns = model.feature_names_in_  # Assumes sklearn model with feature names stored
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match model's training columns
    input_data = input_data[required_columns]

    # Make prediction and get probability
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]
    
    return prediction[0], probability[0]

# Streamlit App Interface
st.title("Heart Disease Prediction App")
st.write("Input the following health metrics to predict the likelihood of heart disease.")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=["male", "female"])
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])  # Modify based on dataset
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])  # Modify based on dataset

# Map user input to model input format
features = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

# Predict button
if st.button("Predict"):
    # Preprocess input features and make prediction
    prediction, probability = preprocess_and_predict(features)

    # Display the result
    if prediction == 1:
        st.error(f"There is a high chance of heart disease with a probability of {probability:.2f}.")
    else:
        st.success(f"There is a low chance of heart disease with a probability of {probability:.2f}.")

    st.write("Note: This prediction is for educational purposes only and not a substitute for professional medical advice.")


# In[ ]:




