from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "churn_ann.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900)
    Geography: str = Field(..., pattern="^(France|Germany|Spain)$")
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/model-info")
def model_info():
    return metrics

@app.post("/api/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained. Run backend/train_model.py first.")
    payload = customer.model_dump()
    probability = float(model.predict_proba(pd.DataFrame([payload]))[0][1])
    prediction = int(probability >= 0.5)
    risk = "High" if probability >= 0.60 else "Medium" if probability >= 0.30 else "Low"
    return {
        "churn_probability": round(probability, 4),
        "churn_percentage": round(probability * 100, 2),
        "prediction": prediction,
        "risk": risk,
        "message": "Customer is predicted to churn." if prediction else "Customer is predicted to remain.",
    }

@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")

@app.get("/app.js")
def app_js():
    return FileResponse(FRONTEND / "app.js", media_type="application/javascript")

@app.get("/styles.css")
def styles():
    return FileResponse(FRONTEND / "styles.css", media_type="text/css")
