
import streamlit as st
import pandas as pd
import joblib
import joblib

model = joblib.load("model.pkl")# Even though joblib was used, load_model is safer for Keras

# Load the trained model and scaler
try:
    # It's generally recommended to save Keras models with model.save() and load with load_model()
    # However, since you used joblib.dump, we'll try joblib.load for consistency.
    # If this fails, consider saving the model with `model.save('model.h5')` instead.
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    st.error("Error: model.joblib or scaler.joblib not found. Please ensure they are in the same directory as app.py.")
    st.stop()

st.title("Insurance Charges Prediction App")
st.write("Enter the client's details to predict their insurance charges.")

# Input fields for user
age = st.slider("Age", 18, 64, 30)
sex = st.radio("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.radio("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Create a dictionary from inputs
input_data = {
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region': region
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocessing steps (matching the training data)
# 1. One-hot encoding
input_df_encoded = pd.get_dummies(input_df, columns=["sex", "smoker", "region"], drop_first=True)

# Define the expected columns from the training data
# This ensures that the input to the model always has the same features in the same order
expected_columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes',
                    'region_northwest', 'region_southeast', 'region_southwest']

# Add missing columns with a default value (False for dummy variables)
for col in expected_columns:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = False

# Reorder columns to match the training data's order
input_df_encoded = input_df_encoded[expected_columns]

# 2. Scaling
input_scaled = scaler.transform(input_df_encoded)

# Make prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_scaled)[0][0]
    st.success(f"Predicted Insurance Charges: ${prediction:.2f}")
