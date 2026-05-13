import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from feature_engineering import prepare_tfidf_data
import joblib


# -----------------------------
# Load Data
# -----------------------------
data_path = "../data/processed/processed_jobs.csv"
df = pd.read_csv(data_path)

X_train, X_test, y_train, y_test, tfidf = prepare_tfidf_data(df)

# -----------------------------
# Logistic Regression (Baseline 1)
# -----------------------------
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # IMPORTANT for imbalance
    n_jobs=-1
)

lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print(confusion_matrix(y_test, lr_preds))
print(classification_report(y_test, lr_preds))
print("ROC-AUC:", roc_auc_score(y_test, lr_probs))

# Save model
joblib.dump(lr, "../models/tfidf/logistic_regression.pkl")
joblib.dump(tfidf, "../models/tfidf/tfidf_vectorizer.pkl")

# -----------------------------
# Random Forest (Baseline 2)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

print("\n=== Random Forest ===")
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

# Save model
joblib.dump(rf, "../models/tfidf/random_forest.pkl")
