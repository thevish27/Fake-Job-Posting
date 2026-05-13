import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import numpy as np
import joblib

# -----------------------------
# Load model & tokenizer
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertModel.from_pretrained(MODEL_NAME)
model.eval()   # inference mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# -----------------------------
# Generate embeddings
# -----------------------------
def generate_embeddings(texts, max_length=256):
    embeddings = []

    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # CLS token embedding
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embedding[0])

    return np.array(embeddings)


if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv("../data/processed/processed_jobs.csv")

    print("Generating DistilBERT embeddings...")
    X_embeddings = generate_embeddings(df["full_text"].tolist())

    # Save embeddings
    np.save("../models/distilbert/distilbert_embeddings.npy", X_embeddings)

    # Save labels
    y = df["fraudulent"].astype(str).str.lower().map({"f": 0, "t": 1})
    np.save("../models/distilbert/labels.npy", y.values)

    print("Embeddings saved successfully")
    print("Shape:", X_embeddings.shape)
