import pandas as pd
import numpy as np
import re
from textblob import TextBlob

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Statistical Text Features
# -----------------------------
def extract_text_features(text):
    if not text:
        return {
            "text_length": 0,
            "word_count": 0,
            "sentence_count": 0,
            "uppercase_ratio": 0,
            "special_char_ratio": 0,
            "sentiment": 0
        }

    words = text.split()
    sentences = text.split('.')
    uppercase_chars = sum(1 for c in text if c.isupper())
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())

    return {
        "text_length": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "uppercase_ratio": uppercase_chars / len(text),
        "special_char_ratio": special_chars / len(text),
        "sentiment": TextBlob(text).sentiment.polarity
    }


# -----------------------------
# Main Preprocessing Function
# -----------------------------
def preprocess_data(df):
    # Combine text fields
    text_columns = [
        'title',
        'company_profile',
        'description',
        'requirements',
        'benefits'
    ]

    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    df['full_text'] = df[text_columns].apply(
        lambda row: ' '.join(row.values.astype(str)),
        axis=1
    )

    # Extract statistical features
    text_features = df['full_text'].apply(extract_text_features)
    text_features_df = pd.DataFrame(text_features.tolist())

    # Binary columns
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in binary_cols:
        df[col] = df[col].astype(str).str.lower().map({
            't': 1, 'f': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0
        }).fillna(0)

    # Categorical columns
    categorical_cols = [
        'employment_type',
        'required_experience',
        'required_education',
        'industry',
        'function'
    ]

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Final dataframe
    processed_df = pd.concat(
        [df[['full_text', 'fraudulent'] + binary_cols + categorical_cols],
         text_features_df],
        axis=1
    )

    return processed_df
