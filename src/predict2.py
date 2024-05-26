import pandas as pd
import joblib

def make_prediction(input_data):
    model = joblib.load('D:\\WhalePredict\\model\\whale_spotting_model.pkl')
    scaler = joblib.load('D:\\WhalePredict\\model\\scaler.pkl')


    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    return prediction, prediction_proba

if __name__ == "__main__":
    input_data = {
        'Temperature': 35,
        'Sea_State': 7,
        'Wind_Speed': 20,
        'Weather_Prediction_Days': 2,
        'Month': 3,
        'Day': 3,
        'Day_Of_Week': 3
    }
    prediction, prediction_proba = make_prediction(input_data)
    print(f"Prediction: {prediction[0]}, Probability: {prediction_proba[0][1]}")
