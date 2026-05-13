import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from feature_engineering import prepare_tfidf_data

# -----------------------------
# Load data
# -----------------------------
data_path = "../data/processed/processed_jobs.csv"
df = pd.read_csv(data_path)

X_train, X_test, y_train, y_test, tfidf = prepare_tfidf_data(df)

# Load trained Logistic Regression
lr = joblib.load("../models/tfidf/logistic_regression.pkl")

# Get probabilities
y_probs = lr.predict_proba(X_test)[:, 1]

# -----------------------------
# Threshold tuning
# -----------------------------
thresholds = np.arange(0.1, 0.9, 0.05)

print("Threshold | Precision | Recall | F1-score")
print("------------------------------------------")

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{t:.2f}      | {precision:.2f}      | {recall:.2f}  | {f1:.2f}")
