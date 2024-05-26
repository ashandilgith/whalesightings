import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
    # Additional preprocessing steps
    return df

if __name__ == "__main__":
    df = load_data('D:/WhalePredict/data/raw/synthetic_whale_sightings.csv')
    df = preprocess_data(df)
    df.to_csv('D:/WhalePredict/data/processed/whale_sightings_processed.csv', index=False)

