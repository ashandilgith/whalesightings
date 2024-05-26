import streamlit as st
from predict import make_prediction




st.title("Whale Spotting Prediction App")
st.write("This app predicts the probability of seeing a whale on a given day based on environmental conditions.")

temperature = st.number_input("Temperature (C)", min_value=5, max_value=45)
sea_state = st.slider("Sea State (0-10)", min_value=0, max_value=10)
wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=100)
weather_prediction_days = st.slider("Days from Weather Prediction", min_value=0, max_value=7)
month = st.slider("Month", min_value=1, max_value=12)
day = st.slider("Day of the Month", min_value=1, max_value=31)
day_of_week = st.slider("Day of the Week", min_value=0, max_value=6)

if st.button("Predict"):
    input_data = {
        'Temperature': temperature,
        'Sea_State': sea_state,
        'Wind_Speed': wind_speed,
        'Weather_Prediction_Days': weather_prediction_days,
        'Month': month,
        'Day': day,
        'Day_Of_Week': day_of_week
    }
    prediction, prediction_proba = make_prediction(input_data)
    st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Probability of seeing a whale: {prediction_proba[0][1]:.2f}")
