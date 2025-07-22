# src/data_preprocessing.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data():
    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    target = housing.target_names[0]

    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    data_dir = 'data/processed'
    os.makedirs(data_dir, exist_ok=True)

    X_train_scaled_df.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)

    print(f"Processed data saved to {data_dir}/")
    print(f"X_train shape: {X_train_scaled_df.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_scaled_df.shape}, y_test shape: {y_test.shape}")

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler

if __name__ == "__main__":
    load_and_preprocess_data()
