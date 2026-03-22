import numpy as np

def F4_summary(y_pred, y_true):
    """
    Module 4: Summary

    Input:
        y_pred : predicted values
        y_true : actual values

    Output:
        accuracy_percentage
    """

    same_sign = np.sign(y_pred) == np.sign(y_true)

    accuracy = np.mean(same_sign) * 100

    return accuracy
