import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from feature_engineering import prepare_tfidf_data

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/processed/processed_jobs.csv")

X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_tfidf_data(df)

X_bert = np.load("../models/distilbert/distilbert_embeddings.npy")

# Recreate same split
X_train_bert, X_test_bert, _, _ = train_test_split(
    X_bert,
    y_train.append(y_test).values,
    test_size=0.2,
    random_state=42,
    stratify=y_train.append(y_test).values
)

# Fuse features
X_test = hstack([X_test_tfidf, X_test_bert])

# Load hybrid model
model = joblib.load("../models/hybrid_logistic_regression.pkl")

# Get probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Threshold tuning
# -----------------------------
print("Threshold | Precision | Recall | F1")
print("----------------------------------")

for t in np.arange(0.3, 0.85, 0.05):
    y_pred = (y_probs >= t).astype(int)
    print(
        f"{t:.2f}      | "
        f"{precision_score(y_test, y_pred):.2f}      | "
        f"{recall_score(y_test, y_pred):.2f}  | "
        f"{f1_score(y_test, y_pred):.2f}"
    )
