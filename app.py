import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("crop_yield_model.pkl")

# Title
st.title("🌾 Crop Yield Prediction App")
st.write("This app predicts the **crop yield (tons/hectare)** based on input features.")

# Input form
with st.form("prediction_form"):
    crop = st.text_input("Crop (e.g., Rice, Maize)")
    crop_year = st.number_input("Crop Year", min_value=1990, max_value=2100, value=2022)
    season = st.text_input("Season (e.g., Kharif, Rabi, Whole Year)")
    state = st.text_input("State (e.g., Karnataka, Punjab)")
    area = st.number_input("Area (hectares)", min_value=0.0, value=1.0)
    production = st.number_input("Production (tons)", min_value=0.0, value=1.0)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0)
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0.0, value=50.0)
    pesticide = st.number_input("Pesticide Used (kg/ha)", min_value=0.0, value=5.0)

    submitted = st.form_submit_button("Predict Yield")

if submitted:
    # Preprocess (you must match the label encoding used in training)
    try:
        input_df = pd.DataFrame({
            'Crop': [crop],
            'Crop_Year': [crop_year],
            'Season': [season],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })

        # Encode categorical features if needed (you should save and load LabelEncoders used in training)
        # For now, we assume the model was trained on already encoded data or string labels
        # So this will only work if your training data handled string categories internally in pipeline

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/hectare")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
