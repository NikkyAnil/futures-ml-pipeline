# module4_evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(y_true, y_pred):
    """
    Computes metrics + trading strategy
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    accuracy = np.mean(direction_true == direction_pred)

    # Trading strategy
    capital = 10000
    capital_history = [capital]

    for i in range(len(y_pred)):
        if y_pred[i] > 0:
            capital += y_true[i]
        else:
            capital -= y_true[i]

        capital_history.append(capital)

    return {
        "mse": mse,
        "mae": mae,
        "directional_accuracy": accuracy,
        "final_capital": capital,
        "capital_history": capital_history
    }
