import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

from feature_engineering import prepare_tfidf_data

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv("../data/processed/processed_jobs.csv")

# TF-IDF features
X_tfidf, _, y, tfidf = None, None, None, None
X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_tfidf_data(df)

# DistilBERT embeddings
X_bert = np.load("../models/distilbert/distilbert_embeddings.npy")

# Stratified split indices (same split)
X_train_bert, X_test_bert, _, _ = train_test_split(
    X_bert,
    y_train.append(y_test).values,
    test_size=0.2,
    random_state=42,
    stratify=y_train.append(y_test).values
)

# -----------------------------
# Feature Fusion
# -----------------------------
X_train = hstack([X_train_tfidf, X_train_bert])
X_test = hstack([X_test_tfidf, X_test_bert])

print("Hybrid train shape:", X_train.shape)
print("Hybrid test shape:", X_test.shape)

# -----------------------------
# Train Hybrid Model
# -----------------------------
hybrid_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

hybrid_lr.fit(X_train, y_train)

y_preds = hybrid_lr.predict(X_test)
y_probs = hybrid_lr.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
print("\n=== HYBRID MODEL (TF-IDF + DistilBERT) ===")
print(confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))
print("ROC-AUC:", roc_auc_score(y_test, y_probs))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(hybrid_lr, "../models/hybrid_logistic_regression.pkl")
joblib.dump(tfidf, "../models/hybrid_tfidf_vectorizer.pkl")
