# run_pipeline.py

import sys
import matplotlib.pyplot as plt

from module1_dataprep import prepare_data
from module2_training import train_model
from module3_testing import test_model
from module4_evaluation import evaluate_model

# Inputs from CLI / GUI
file_path = sys.argv[1]
lookback = int(sys.argv[2])

print("Running pipeline...")

# Step 1: Data Prep
X, y = prepare_data(file_path, lookback=lookback)

# Split data
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 2: Training
model, scaler = train_model(X_train, y_train)

# Step 3: Testing
predictions, y_test = test_model(model, scaler, X_test, y_test)

# Step 4: Evaluation
results = evaluate_model(y_test, predictions)

# Print results
print("MSE:", results["mse"])
print("MAE:", results["mae"])
print("Directional Accuracy:", results["directional_accuracy"])
print("Final Capital:", results["final_capital"])

# Plot equity curve
plt.plot(results["capital_history"])
plt.title("Money Under Management")
plt.xlabel("Time")
plt.ylabel("Capital")
plt.show()
