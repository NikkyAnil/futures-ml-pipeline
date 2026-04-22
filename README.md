Futures ML Pipeline
A modular machine learning pipeline for predicting minute-level futures price direction using SVM and Gradient Boosting.

What this does
Takes minute-level futures data (TY futures CSV), trains a classification model to predict whether the next minute's price will go up or down, and evaluates performance using accuracy, ROC curve, and a simulated trading equity curve.

Directory structure
futures-ml-pipeline/
│
├── module1_dataprep/
│   └── dataprep.py          ← Loads CSV, builds feature matrix
│
├── module2_training/
│   └── training.py          ← Trains SVM or Gradient Boost model
│
├── module3_testing/
│   └── testing.py           ← Generates predictions on test data
│
├── module4_statistics/
│   └── statistics.py        ← Computes metrics and plots charts
│
├── run_pipeline.py           ← Runs all 4 modules automatically
├── gui_app.py                ← Web GUI (open in browser)
├── TY.csv                    ← Your data file (place here)
└── README.md

Installation
Install required Python packages:
bashpip install flask scikit-learn pandas numpy matplotlib

How to run — Option 1: Command line (fully automatic)
Run the complete pipeline with one command:
bashpython run_pipeline.py
This loads TY.csv, trains the model, tests it, and saves a results chart.
Available options:
bashpython run_pipeline.py --csv TY.csv --model svm --max_rows 5000
OptionWhat it doesDefault--csvPath to your CSV data fileTY.csv--t2Date that splits training and test data2023-10-01--lookbackHow many minutes to look back for features4--max_rowsMaximum rows to load (limits runtime)5000--modelChoose svm or gradient_boostsvm--thresholdConfidence needed to count a trade0.55--use_relative1 = include % change features, 0 = absolute only1--train_as_test1 = use training data as test (overfit check)0
Examples:
Run with Gradient Boosting:
bashpython run_pipeline.py --model gradient_boost
Run with train = test (checks if model can learn at all):
bashpython run_pipeline.py --train_as_test 1
Run on small toy dataset (100 rows):
bashpython run_pipeline.py --max_rows 100
Run without relative features (absolute prices only):
bashpython run_pipeline.py --use_relative 0

How to run — Option 2: Web GUI (recommended for demos)
Start the GUI server:
bashpython gui_app.py
Then open your browser at:
http://localhost:5000
The GUI lets you:

Choose your CSV file
Set training and test date ranges (T1, T2, T3)
Pick SVM or Gradient Boosting
Toggle relative features on/off
Toggle train-as-test mode on/off
Watch the pipeline run live with a step-by-step progress indicator
See results instantly: accuracy, precision, ROC AUC, P&L, and equity chart


Module descriptions
Module 1 — Data Preparation (module1_dataprep/dataprep.py)
What it does:
Reads the CSV file and converts raw minute-level price data into a supervised learning matrix using a rolling lookback window.
Input: CSV file path, lookback window size, relative features flag, max rows
Output: Feature matrix X, target vector y, feature names, timestamps
Features created:

For each of the previous N minutes: Open, High, Low, Close, VWAP, Volume, Up Ticks, Down Ticks, Same Ticks (9 features × N minutes = absolute features)
Same values expressed as percentage change from the opening price of the window (relative features)
Target y = percentage price change of the next minute: (close - open) / open × 100

LLM prompt used:

"I have minute-level futures data with columns: Date, Time, Open, High, Low, Close, VWAP, Volume, Up Ticks, Down Ticks, Same Ticks. Convert it into supervised learning format using a rolling lookback window of N minutes. For each window produce absolute OHLCV feature values and relative percentage change of every value vs the opening price of the window. Target y = percentage change of next minute: (close - open) / open × 100. Include a flag use_relative to toggle relative features on/off."


Module 2 — Training (module2_training/training.py)
What it does:
Scales the features and trains a binary classification model. The continuous target y is converted to binary labels: +1 if price went up, -1 if price went down.
Input: X_train, y_train, model type string
Output: Trained model object, fitted StandardScaler
Supported models:

svm — Support Vector Classifier with RBF kernel
gradient_boost — Gradient Boosting Classifier (100 trees, depth 3)

LLM prompt used:

"Train a binary classifier on financial time-series features. Convert continuous target y (percentage price change) to binary labels: +1 if price went up (y > 0), -1 if price went down (y <= 0). Support two model types selectable by a string argument: svm (SVC with RBF kernel, probability=True) and gradient_boost (GradientBoostingClassifier). Scale features with StandardScaler before training. Return the trained model and the fitted scaler."


Module 3 — Testing (module3_testing/testing.py)
What it does:
Applies the trained model to unseen test data. Uses the same scaler fitted during training to avoid data leakage.
Input: Trained model, fitted scaler, X_test, y_test
Output: y_true (binary ground truth), y_pred (predicted labels), y_proba (confidence probabilities for ROC curve)
LLM prompt used:

"Use a trained sklearn classifier to predict on unseen test data. Apply the same StandardScaler that was fitted during training. Return y_true (binary ground-truth labels +1/-1 derived from y_test), y_pred (predicted binary labels), y_proba (probability of the positive class for ROC curve)."


Module 4 — Statistics (module4_statistics/statistics.py)
What it does:
Evaluates model performance and generates a 4-panel chart saved as results.png.
Input: y_true, y_pred, y_proba, y_continuous (for AUM), threshold
Output: Metrics dictionary, AUM curve, ROC curve data, saved figure
Metrics computed:

Overall directional accuracy (% of predictions with correct direction)
Positive precision: of all times we predicted "up", how many were right
Negative precision: of all times we predicted "down", how many were right
Threshold accuracy: accuracy counting only trades with confidence ≥ threshold
AUM equity curve: $10,000 starting capital, long if +1 prediction, short if -1
ROC curve and AUC score

LLM prompt used:

"Create a statistics module for a binary trading classifier. Compute: overall directional accuracy, precision for + predictions (long calls) and - predictions (short calls) separately, accuracy above a confidence threshold, AUM equity curve starting at $10,000, ROC curve and AUC. Plot everything on a clean black-on-white 2×2 matplotlib figure."


Understanding the results
MetricWhat it meansTargetOverall accuracy% of minutes where direction was predicted correctly> 55%+ PrecisionOf times model said "price up", how often was it right> 50%− PrecisionOf times model said "price down", how often was it right> 50%ROC AUC0.5 = random guessing, 1.0 = perfect> 0.55P&LProfit or loss from $10,000 starting capitalPositive
Important: If overall accuracy is below 50%, the model is worse than random guessing. Use --train_as_test 1 to check if the model can learn at all on training data. If accuracy is still low on training data, there is a bug in the pipeline.

Data format
The CSV file must have these columns:
ColumnDescriptionDate and TimeTimestamp (e.g. 01/02/2023 17:05)OpenOpening price of the minuteHighHighest price of the minuteLowLowest price of the minuteCloseClosing price of the minuteVWAPVolume weighted average priceVolumeNumber of contracts tradedUp TicksNumber of up-tick tradesDown TicksNumber of down-tick tradesSame TicksNumber of same-price trades

GitHub workflow
Each team member should work on their own branch:
bashgit checkout -b nick/pipeline-updates
git add .
git commit -m "Add relative features to module1"
git push origin nick/pipeline-updates
Then open a pull request to merge into main.
