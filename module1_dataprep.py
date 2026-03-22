import pandas as pd
import numpy as np

def F1_dataprep(csv_file):
    """
    Module 1: Data Preparation
    
    Input:
        csv_file : path to dataset CSV
        
    Output:
        X : feature matrix (36 features per sample)
        y : target vector (close - open for next minute)
    """

    # Load dataset
    df = pd.read_csv(csv_file)

    # Features expected in dataset
    features = [
        "Open",
        "Low",
        "High",
        "Close",
        "VWAP",
        "Volume",
        "UpTicks",
        "DownTicks",
        "SameTicks"
    ]

    # Check features exist
    df = df[features]

    X = []
    y = []

    # Build training samples
    for i in range(4, len(df) - 1):

        # Previous 4 minutes features
        window = df.iloc[i-4:i].values.flatten()

        # Target = next minute price change
        next_row = df.iloc[i]
        target = next_row["Close"] - next_row["Open"]

        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    return X, y
