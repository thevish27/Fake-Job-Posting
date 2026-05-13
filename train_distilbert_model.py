import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

# -----------------------------
# Load embeddings & labels
# -----------------------------
X = np.load("../models/distilbert/distilbert_embeddings.npy")
y = np.load("../models/distilbert/labels.npy")

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)

# -----------------------------
# Train-test split (STRATIFIED)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Logistic Regression on embeddings
# -----------------------------
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

lr.fit(X_train, y_train)

y_preds = lr.predict(X_test)
y_probs = lr.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
print("\n=== DistilBERT + Logistic Regression ===")
print(confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(lr, "../models/distilbert/logistic_regression_distilbert.pkl")
