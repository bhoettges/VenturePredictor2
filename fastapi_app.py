from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import Tool as LC_Tool

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

class ChatRequest(BaseModel):
    message: str

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

# --- LangChain tools for chat agent ---
def parse_features(input_str):
    try:
        features = [float(x.strip()) for x in input_str.split(',')]
        return features
    except Exception:
        return None

def predict_arr_growth_xgb(features):
    X = np.array([features])
    prediction = model_xgb.predict(X)
    return prediction.tolist()

def predict_arr_growth_rf(features):
    X = np.array([features])
    prediction = model_rf.predict(X)
    return prediction.tolist()

def arr_tool_xgb(input_str):
    features = parse_features(input_str)
    if features is None or len(features) != 28:
        return "Please provide exactly 28 comma-separated features (7 per quarter for 4 quarters)."
    result = predict_arr_growth_xgb(features)
    return f"[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}"

def arr_tool_rf(input_str):
    features = parse_features(input_str)
    if features is None or len(features) != 28:
        return "Please provide exactly 28 comma-separated features (7 per quarter for 4 quarters)."
    result = predict_arr_growth_rf(features)
    return f"[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}"

def csv_tool(input_str, model_choice='xgboost'):
    try:
        if os.path.exists(input_str):
            df = pd.read_csv(input_str)
        else:
            from io import StringIO
            df = pd.read_csv(StringIO(input_str))
    except Exception as e:
        return f"Could not read CSV: {e}"
    try:
        if len(df) < 4:
            return "CSV must have at least 4 rows (quarters)."
        df_last4 = df.tail(4)
        features = []
        for _, row in df_last4.iterrows():
            for field in required_fields:
                features.append(float(row[field]))
        if len(features) != 28:
            return "CSV does not contain all required fields for 4 quarters."
        if model_choice == 'random_forest':
            result = predict_arr_growth_rf(features)
            return f"[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}"
        else:
            result = predict_arr_growth_xgb(features)
            return f"[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}"
    except Exception as e:
        return f"Error processing CSV: {e}"

arr_growth_tool_xgb = LC_Tool(
    name="XGBoost ARR Growth Predictor",
    func=arr_tool_xgb,
    description="Predicts ARR YoY growth for the next 4 quarters using the XGBoost multi-output model. Input: 28 comma-separated features."
)
arr_growth_tool_rf = LC_Tool(
    name="Random Forest ARR Growth Predictor",
    func=arr_tool_rf,
    description="Predicts ARR YoY growth for the next 4 quarters using the Random Forest multi-output model. Input: 28 comma-separated features."
)
csv_growth_tool_xgb = LC_Tool(
    name="XGBoost CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, model_choice='xgboost'),
    description="Predicts ARR YoY growth for the next 4 quarters using XGBoost from a CSV file or CSV string."
)
csv_growth_tool_rf = LC_Tool(
    name="Random Forest CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, model_choice='random_forest'),
    description="Predicts ARR YoY growth for the next 4 quarters using Random Forest from a CSV file or CSV string."
)

llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = initialize_agent(
    tools=[arr_growth_tool_xgb, arr_growth_tool_rf, csv_growth_tool_xgb, csv_growth_tool_rf],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    message = request.message.lower()
    # Keywords for specialized agent
    keywords = ["arr", "growth", "predict", "csv", "random forest", "xgboost"]
    if any(kw in message for kw in keywords):
        try:
            response = agent.run(request.message)
        except Exception as e:
            response = f"Sorry, there was an error: {str(e)}"
    else:
        # Use the LLM directly for general chat, with a friendly system prompt
        system_prompt = (
            "Hi! I'm your venture prediction assistant. "
            "Would you like to predict your future ARR using a machine learning model, "
            "or would you like to chat about something else? "
            "If you want a prediction, just tell me your data or upload a CSV!"
        )
        full_prompt = f"{system_prompt}\n\nUser: {request.message}"
        response = llm.invoke(full_prompt).content
    return {"response": response}

@app.get("/")
def root():
    return {"message": "ARR Growth Prediction API. Use /predict_raw, /predict_csv, or /chat."} 