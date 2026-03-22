from sklearn.svm import SVR

def F2_training(X_train, y_train):
    """
    Module 2: Training

    Input:
        X_train : feature matrix
        y_train : target vector

    Output:
        trained_model : trained SVM model
    """

    model = SVR(kernel="linear")

    model.fit(X_train, y_train)

    return model
