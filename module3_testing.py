def F3_testing(model, X_test):
    """
    Module 3: Testing

    Input:
        model : trained ML model
        X_test : feature matrix for test data

    Output:
        y_pred : predicted y values
    """

    y_pred = model.predict(X_test)

    return y_pred
