import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    df = pd.read_csv('D:/WhalePredict/data/raw/synthetic_whale_sightings.csv')
    X = df.drop('Whale_Sighting', axis=1)
    y = df['Whale_Sighting']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'D:/WhalePredict/model/whale_spotting_model.pkl')
    joblib.dump(scaler, 'D:/WhalePredict/model/scaler.pkl')

if __name__ == "__main__":
    train_model()
