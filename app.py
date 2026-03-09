import streamlit as st
import joblib
import numpy as np
from datetime import date

# Load saved model
model = joblib.load("model.joblib")

#Date
today=date.today()
st.write(today)

# App Title
st.title("🏠 House Price Prediction App")

st.write("Enter house area to predict price")

# User Input
area = st.number_input("Enter Area (sq ft)", min_value=500, max_value=5000, value=1000)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(np.array([[area]]))
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")