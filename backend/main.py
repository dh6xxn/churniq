from pathlib import Path
import io, json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "churn_ann.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
FRONTEND = ROOT / "frontend"

FEATURES = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]
IDENTIFIERS = ["CustomerId","Surname","RowNumber"]

app = FastAPI(title="ChurnIQ API", version="2.0.0")
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

def risk_band(p):
    return "High" if p >= .60 else "Medium" if p >= .30 else "Low"

def predict_frame(df):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained.")
    probs = model.predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["ChurnProbability"] = probs.round(6)
    out["ChurnPercentage"] = (probs * 100).round(2)
    out["Prediction"] = (probs >= .50).astype(int)
    out["Risk"] = [risk_band(p) for p in probs]
    return out.sort_values("ChurnProbability", ascending=False)

def validate_csv(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message":"Missing required columns.","missing_columns":missing})
    work = df.copy()
    errors = []
    for c in FEATURES:
        if c not in ["Geography","Gender"]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    valid = (
        work["CreditScore"].between(300,900) &
        work["Age"].between(18,100) &
        work["Tenure"].between(0,10) &
        (work["Balance"] >= 0) &
        work["NumOfProducts"].between(1,4) &
        work["HasCrCard"].isin([0,1]) &
        work["IsActiveMember"].isin([0,1]) &
        (work["EstimatedSalary"] >= 0) &
        work["Geography"].isin(["France","Germany","Spain"]) &
        work["Gender"].isin(["Male","Female"])
    )
    invalid = int((~valid).sum())
    if invalid:
        errors.append(f"{invalid} row(s) failed validation and were excluded.")
    return work.loc[valid].copy(), errors, invalid

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":model is not None}

@app.get("/api/model-info")
def model_info():
    return metrics

@app.post("/api/predict")
def predict(customer: Customer):
    result = predict_frame(pd.DataFrame([customer.model_dump()])).iloc[0]
    return {
        "churn_probability": float(result["ChurnProbability"]),
        "churn_percentage": float(result["ChurnPercentage"]),
        "prediction": int(result["Prediction"]),
        "risk": result["Risk"],
        "message": "Customer is predicted to churn." if int(result["Prediction"]) else "Customer is predicted to remain."
    }

@app.post("/api/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")
    valid, errors, invalid = validate_csv(df)
    if valid.empty:
        raise HTTPException(status_code=422, detail={"message":"No valid rows found.","errors":errors})
    result = predict_frame(valid)
    preview = result.head(100).where(pd.notna(result.head(100)), None).to_dict(orient="records")
    return {
        "filename": file.filename,
        "total_rows": int(len(df)),
        "valid_rows": int(len(valid)),
        "invalid_rows": invalid,
        "errors": errors,
        "predicted_churners": int(result["Prediction"].sum()),
        "predicted_non_churners": int((result["Prediction"] == 0).sum()),
        "high_risk": int((result["Risk"] == "High").sum()),
        "medium_risk": int((result["Risk"] == "Medium").sum()),
        "low_risk": int((result["Risk"] == "Low").sum()),
        "average_churn_probability": round(float(result["ChurnProbability"].mean()), 4),
        "rows": preview
    }

@app.post("/api/predict-csv/download")
async def predict_csv_download(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    df = pd.read_csv(io.BytesIO(await file.read()))
    valid, errors, invalid = validate_csv(df)
    if valid.empty:
        raise HTTPException(status_code=422, detail={"message":"No valid rows found.","errors":errors})
    result = predict_frame(valid)
    buf = io.StringIO()
    result.to_csv(buf, index=False)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=churn_predictions.csv"})

@app.get("/")
def index(): return FileResponse(FRONTEND / "index.html")
@app.get("/app.js")
def app_js(): return FileResponse(FRONTEND / "app.js", media_type="application/javascript")
@app.get("/styles.css")
def styles(): return FileResponse(FRONTEND / "styles.css", media_type="text/css")
