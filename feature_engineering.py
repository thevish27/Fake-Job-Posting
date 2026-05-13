import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib


def tfidf_features(df, max_features=5000):
    """
    Generates TF-IDF features from full_text
    """

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9
    )

    X_tfidf = tfidf.fit_transform(df['full_text'])
    y = df['fraudulent'].astype(str).str.lower().map({'f': 0, 't': 1})


    return X_tfidf, y, tfidf


def prepare_tfidf_data(df):
    """
    Prepares train-test split with stratification
    """

    X_tfidf, y, tfidf = tfidf_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y   # VERY IMPORTANT for imbalance
    )

    return X_train, X_test, y_train, y_test, tfidf
