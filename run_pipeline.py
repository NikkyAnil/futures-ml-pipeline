from dataprep_module import F1_dataprep
from training_module import F2_training
from testing_module import F3_testing
from summary_module import F4_summary

# Step 1: Prepare data
X, y = F1_dataprep("TY-2023_2026.csv")

# Split into training and testing
split_index = int(0.8 * len(X))

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]

# Step 2: Train model
model = F2_training(X_train, y_train)

# Step 3: Predict
y_pred = F3_testing(model, X_test)

# Step 4: Evaluate
accuracy = F4_summary(y_pred, y_test)

print("Prediction sign accuracy:", accuracy, "%")
