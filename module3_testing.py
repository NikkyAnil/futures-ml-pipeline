# module3_testing.py

def test_model(model, scaler, X_test, y_test):
    """
    Applies model on test data (no leakage)
    """

    X_scaled = scaler.transform(X_test)

    predictions = model.predict(X_scaled)

    return predictions, y_test
