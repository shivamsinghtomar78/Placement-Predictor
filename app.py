import streamlit as st
import pickle
import numpy as np

 
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

 
st.title("Student Placement Prediction")

 
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0)
iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, value=100.0)
 
if st.button("Predict"):
 
    input_data = scaler.transform(np.array([[cgpa, iq]]))
 
    prediction = model.predict(input_data)
 
    if prediction[0] == 1:
        st.success("The student is likely to be placed.")
    else:
        st.error("The student is unlikely to be placed.")