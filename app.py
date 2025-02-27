import streamlit as st
import pickle
import numpy as np
import time  # Import the time module for animation

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# --- Animated Title ---
st.markdown(
    """
    <style>
    .animated-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        animation: color-change 5s infinite;
    }
    @keyframes color-change {
        0% { color: #FF5733; }
        25% { color: #33FF57; }
        50% { color: #3357FF; }
        75% { color: #FF33E9; }
        100% { color: #FF5733; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='animated-title'>Student Placement Predictor</div>", unsafe_allow_html=True)

# --- Input Area with Expansion ---
with st.expander("Enter Student Details Here"):
    cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    iq = st.number_input("Enter IQ", min_value=0.0, max_value=200.0, value=100.0, step=1.0)

# --- Prediction Button with Loading Animation ---
if st.button("Predict Placement"):
    # --- Loading Animation ---
    with st.spinner("Predicting student placement..."):
        bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)  # Simulate some processing time
            bar.progress(i + 1)

        # --- Prediction ---
        input_data = scaler.transform(np.array([[cgpa, iq]]))
        prediction = model.predict(input_data)

    # --- Animated Prediction Result ---
    if prediction[0] == 1:
        # Success Animation (Removed balloons)
        st.success("🎉 The student is likely to be placed! 🎉")
    else:
        # Failure Animation (Shake)
        st.markdown(
        """
        <style>
        .shake {
          animation: shake-animation 0.5s ease-in-out;
          color: red;
        }

        @keyframes shake-animation {
          0% { transform: translateX(0); }
          20% { transform: translateX(-10px); }
          40% { transform: translateX(10px); }
          60% { transform: translateX(-10px); }
          80% { transform: translateX(10px); }
          100% { transform: translateX(0); }
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        st.markdown("<div class='shake'>🚨 The student is unlikely to be placed. 🚨</div>", unsafe_allow_html=True)
