# module2_training.py

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train, C=1.0, epsilon=0.1, kernel='rbf'):
    """
    Trains SVM model with scaling
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    model.fit(X_scaled, y_train)

    return model, scaler
