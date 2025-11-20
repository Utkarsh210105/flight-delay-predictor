import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model & encoders
# -----------------------------
model = pickle.load(open("delay_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("✈️ Aircraft Delay Prediction App")
st.write("Enter flight details below to predict if the flight will be delayed.")

# -----------------------------------------
# Helper: Encode categorical inputs
# -----------------------------------------
def encode_value(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Unknown value → add temporarily
        encoder.classes_ = list(encoder.classes_) + [value]
        return encoder.transform([value])[0]

# -----------------------------------------
# User Inputs
# -----------------------------------------
airline = st.selectbox("Airline", encoders["UniqueCarrier"].classes_)
origin = st.selectbox("Origin Airport", encoders["Origin"].classes_)
dest = st.selectbox("Destination Airport", encoders["Dest"].classes_)

dep_hour = st.number_input("Departure Hour (0-23)", 0, 23, 14)
distance = st.number_input("Distance (miles)", 1, 5000, 1200)

month = st.number_input("Month (1-12)", 1, 12, 6)
day = st.number_input("Day", 1, 31, 15)
day_of_week = st.number_input("Day of Week (1=Mon ... 7=Sun)", 1, 7, 5)

# -----------------------------------------
# Predict Button
# -----------------------------------------
if st.button("Predict Delay"):
    
    # Encode categorical text → numeric
    airline_enc = encode_value(encoders["UniqueCarrier"], airline)
    origin_enc = encode_value(encoders["Origin"], origin)
    dest_enc = encode_value(encoders["Dest"], dest)

    # Create input row
    input_row = pd.DataFrame([{
        "UniqueCarrier": airline_enc,
        "Origin": origin_enc,
        "Dest": dest_enc,
        "DepHour": dep_hour,
        "Distance": distance,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week
    }])

    # Prediction
    pred = model.predict(input_row)[0]
    prob = model.predict_proba(input_row)[0][1]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Delay Probability:** {prob*100:.2f}%")

    if pred == 1:
        st.error("⚠️ The flight is **likely to be DELAYED**.")
    else:
        st.success("✅ The flight is **likely to be ON TIME**.")
