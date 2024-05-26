import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic data
    temperature = np.random.uniform(5, 45, num_samples)
    sea_state = np.random.randint(0, 11, num_samples)
    wind_speed = np.random.uniform(0, 100, num_samples)
    weather_prediction_days = np.random.randint(0, 8, num_samples)
    month = np.random.randint(1, 13, num_samples)
    day = np.random.randint(1, 32, num_samples)
    day_of_week = np.random.randint(0, 7, num_samples)
    
    # Randomly assign whale sightings with a probability based on arbitrary rules
    probability = 0.3 + 0.001 * (temperature - 25) - 0.02 * (sea_state - 5) - 0.01 * (wind_speed - 50)
    probability = np.clip(probability, 0, 1)  # Ensure probability is within [0, 1]
    whale_sighting = np.random.binomial(1, p=probability, size=num_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temperature,
        'Sea_State': sea_state,
        'Wind_Speed': wind_speed,
        'Weather_Prediction_Days': weather_prediction_days,
        'Month': month,
        'Day': day,
        'Day_Of_Week': day_of_week,
        'Whale_Sighting': whale_sighting
    })
    
    return df

if __name__ == "__main__":
    num_samples = 1000
    synthetic_data = generate_synthetic_data(num_samples)
    synthetic_data.to_csv('data/raw/synthetic_whale_sightings.csv', index=False)
    print(f"Generated {num_samples} synthetic data samples and saved to 'data/raw/synthetic_whale_sightings.csv'")
