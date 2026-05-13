from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------
# App initialization
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# CORS (required for React)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Resolve absolute paths safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "tfidf", "logistic_regression.pkl"
)
VECTORIZER_PATH = os.path.join(
    BASE_DIR, "models", "tfidf", "tfidf_vectorizer.pkl"
)

# --------------------------------------------------
# Load model & vectorizer
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

THRESHOLD = 0.45

# --------------------------------------------------
# Request schemas
# --------------------------------------------------
class JobRequest(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str


class UrlRequest(BaseModel):
    url: str


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "API running with TF-IDF Logistic Regression"}

# --------------------------------------------------
# Manual input prediction
# --------------------------------------------------
@app.post("/predict")
def predict_job(job: JobRequest):
    combined_text = " ".join([
        job.title,
        job.company_profile,
        job.description,
        job.requirements,
        job.benefits
    ])

    X = vectorizer.transform([combined_text])
    fake_prob = model.predict_proba(X)[0][1]

    prediction = "FAKE" if fake_prob >= THRESHOLD else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": round(float(fake_prob), 3),
        "threshold": THRESHOLD
    }

# --------------------------------------------------
# URL-based prediction
# --------------------------------------------------
@app.post("/predict-url")
def predict_from_url(data: UrlRequest):
    try:
        response = requests.get(
            data.url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )
    except Exception:
        return {"error": "Unable to fetch URL"}

    soup = BeautifulSoup(response.text, "html.parser")

    # ---- Simple extraction heuristics ----
    title_tag = soup.find("h1")
    paragraph_tags = soup.find_all("p")

    title_text = title_tag.get_text(strip=True) if title_tag else ""
    description_text = " ".join(
        [p.get_text(strip=True) for p in paragraph_tags[:5]]
    )

    combined_text = f"{title_text} {description_text}".strip()

    if not combined_text:
        return {"error": "Could not extract job content from URL"}

    X = vectorizer.transform([combined_text])
    fake_prob = model.predict_proba(X)[0][1]

    prediction = "FAKE" if fake_prob >= THRESHOLD else "REAL"

    return {
        "prediction": prediction,
        "fake_probability": round(float(fake_prob), 3),
        "threshold": THRESHOLD,
        "extracted_title": title_text
    }
