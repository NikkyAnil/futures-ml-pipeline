# module1_dataprep.py

import pandas as pd
import numpy as np

def prepare_data(file_path, lookback=4, target_type="difference"):
    """
    Converts time series into supervised learning format using rolling window.
    """

    df = pd.read_csv(file_path)

    # Combine timestamp if separate
    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    features = ["Open", "High", "Low", "Close"]

    X = []
    y = []

    for i in range(lookback, len(df) - 1):
        window = df.iloc[i - lookback:i]

        # Flatten features
        x_row = window[features].values.flatten()

        # Target calculation
        if target_type == "difference":
            target = df.iloc[i]["Close"] - df.iloc[i]["Open"]
        elif target_type == "return":
            target = (df.iloc[i]["Close"] - df.iloc[i]["Open"]) / df.iloc[i]["Open"]
        else:
            raise ValueError("Invalid target type")

        X.append(x_row)
        y.append(target)

    return np.array(X), np.array(y)
