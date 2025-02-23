import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title of the application
st.title("Student Placement Prediction")

# Input fields for cgpa and iq
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0)
iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, value=100.0)

# Predict button
if st.button("Predict"):
    # Scale the input data
    input_data = scaler.transform(np.array([[cgpa, iq]]))
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.success("The student is likely to be placed.")
    else:
        st.error("The student is unlikely to be placed.")