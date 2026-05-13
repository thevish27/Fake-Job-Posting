import pandas as pd
from feature_engineering import prepare_tfidf_data

# Load processed data
data_path = "../data/processed/processed_jobs.csv"
df = pd.read_csv(data_path)

X_train, X_test, y_train, y_test, tfidf = prepare_tfidf_data(df)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Fake ratio (train):", y_train.mean())
print("Fake ratio (test):", y_test.mean())
