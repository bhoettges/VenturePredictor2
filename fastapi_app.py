from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Load models at startup
XGB_MODEL_PATH = 'xgboost_multi_model.pkl'
RF_MODEL_PATH = 'random_forest_model.pkl'
model_xgb = joblib.load(XGB_MODEL_PATH)
model_rf = joblib.load(RF_MODEL_PATH)

required_fields = [
    "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
    "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
]

class FeatureInput(BaseModel):
    features: List[float]
    model: str = 'xgboost'  # or 'random_forest'

@app.post("/predict_raw")
def predict_raw(data: FeatureInput):
    if len(data.features) != 28:
        return JSONResponse(status_code=400, content={"error": "Exactly 28 features required (7 per quarter for 4 quarters)."})
    X = np.array([data.features])
    if data.model == 'random_forest':
        prediction = model_rf.predict(X)
        model_used = 'Random Forest'
    else:
        prediction = model_xgb.predict(X)
        model_used = 'XGBoost'
    return {"model": model_used, "prediction": prediction.tolist()}

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...), model: str = Form('xgboost')):
    try:
        df = pd.read_csv(file.file)
        if len(df) < 4:
            return JSONResponse(status_code=400, content={"error": "CSV must have at least 4 rows (quarters)."})
        df_last4 = df.tail(4)
        features = []
        for _, row in df_last4.iterrows():
            for field in required_fields:
                features.append(float(row[field]))
        if len(features) != 28:
            return JSONResponse(status_code=400, content={"error": "CSV does not contain all required fields for 4 quarters."})
        X = np.array([features])
        if model == 'random_forest':
            prediction = model_rf.predict(X)
            model_used = 'Random Forest'
        else:
            prediction = model_xgb.predict(X)
            model_used = 'XGBoost'
        return {"model": model_used, "prediction": prediction.tolist()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
def root():
    return {"message": "ARR Growth Prediction API. Use /predict_raw or /predict_csv."} 