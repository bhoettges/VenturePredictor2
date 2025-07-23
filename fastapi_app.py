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
import re
from fastapi.middleware.cors import CORSMiddleware
import json
from gpr_analysis import gpr_traffic_light_analysis, gprh_trend_analysis

load_dotenv()

app = FastAPI()

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
XGB_MODEL_PATH = 'xgboost_multi_model.pkl'
RF_MODEL_PATH = 'random_forest_model.pkl'
model_xgb = joblib.load(XGB_MODEL_PATH)
model_rf = joblib.load(RF_MODEL_PATH)

# --- Info Loading ---
try:
    with open("gpt_info.json") as f:
        GPT_INFO = json.load(f)
except Exception:
    GPT_INFO = {}

REQUIRED_FIELDS = [
    "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
    "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
]

# --- Pydantic Models ---
class FeatureInput(BaseModel):
    features: List[float]
    model: str = 'xgboost'

class ChatRequest(BaseModel):
    message: str
    name: str = None
    preferred_model: str = None
    history: list = None

# --- Utility Functions ---
def parse_features(input_str: str):
    """Parse a comma-separated string of features into a list of floats."""
    try:
        return [float(x.strip()) for x in input_str.split(',')]
    except Exception:
        return None

def predict_arr_growth(model, features):
    X = np.array([features])
    return model.predict(X).tolist()

# --- Tool Functions for LangChain Agent ---
def arr_tool(input_str: str, model_choice: str):
    features = parse_features(input_str)
    if features is None or len(features) != 28:
        return "Please provide exactly 28 comma-separated features (7 per quarter for 4 quarters)."
    if model_choice == 'random_forest':
        result = predict_arr_growth(model_rf, features)
        return f"[Random Forest] Predicted ARR YoY growth for the next 4 quarters: {result}"
    else:
        result = predict_arr_growth(model_xgb, features)
        return f"[XGBoost] Predicted ARR YoY growth for the next 4 quarters: {result}"

def csv_tool(input_str: str, model_choice: str = 'xgboost'):
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
            for field in REQUIRED_FIELDS:
                features.append(float(row[field]))
        if len(features) != 28:
            return "CSV does not contain all required fields for 4 quarters."
        return arr_tool(','.join(map(str, features)), model_choice)
    except Exception as e:
        return f"Error processing CSV: {e}"

# --- LangChain Tools ---
arr_growth_tool_xgb = LC_Tool(
    name="XGBoost ARR Growth Predictor",
    func=lambda s: arr_tool(s, 'xgboost'),
    description="Predicts ARR YoY growth for the next 4 quarters using the XGBoost multi-output model. Input: 28 comma-separated features."
)
arr_growth_tool_rf = LC_Tool(
    name="Random Forest ARR Growth Predictor",
    func=lambda s: arr_tool(s, 'random_forest'),
    description="Predicts ARR YoY growth for the next 4 quarters using the Random Forest multi-output model. Input: 28 comma-separated features."
)
csv_growth_tool_xgb = LC_Tool(
    name="XGBoost CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, 'xgboost'),
    description="Predicts ARR YoY growth for the next 4 quarters using XGBoost from a CSV file or CSV string."
)
csv_growth_tool_rf = LC_Tool(
    name="Random Forest CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, 'random_forest'),
    description="Predicts ARR YoY growth for the next 4 quarters using Random Forest from a CSV file or CSV string."
)

# --- LangChain Agent ---
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = initialize_agent(
    tools=[arr_growth_tool_xgb, arr_growth_tool_rf, csv_growth_tool_xgb, csv_growth_tool_rf],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

# --- API Endpoints ---
@app.post("/predict_raw")
def predict_raw(data: FeatureInput):
    """Predict ARR growth from raw features."""
    if len(data.features) != 28:
        return JSONResponse(status_code=400, content={"error": "Exactly 28 features required (7 per quarter for 4 quarters)."})
    model = model_rf if data.model == 'random_forest' else model_xgb
    prediction = model.predict(np.array([data.features]))
    model_used = 'Random Forest' if data.model == 'random_forest' else 'XGBoost'
    return {"model": model_used, "prediction": prediction.tolist()}

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...), model: str = Form('xgboost')):
    """Predict ARR growth from a CSV file upload."""
    try:
        df = pd.read_csv(file.file)
        if len(df) < 4:
            return JSONResponse(status_code=400, content={"error": "CSV must have at least 4 rows (quarters)."})
        df_last4 = df.tail(4)
        features = []
        for _, row in df_last4.iterrows():
            for field in REQUIRED_FIELDS:
                features.append(float(row[field]))
        if len(features) != 28:
            return JSONResponse(status_code=400, content={"error": "CSV does not contain all required fields for 4 quarters."})
        model_obj = model_rf if model == 'random_forest' else model_xgb
        prediction = model_obj.predict(np.array([features]))
        model_used = 'Random Forest' if model == 'random_forest' else 'XGBoost'
        return {"model": model_used, "prediction": prediction.tolist()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Conversational endpoint for GPT agent with prediction and project info."""
    message = request.message.lower()
    name = request.name
    preferred_model = request.preferred_model
    history = request.history or []

    feature_numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", request.message)
    is_data_provided = len(feature_numbers) == 28 or "csv" in message

    if preferred_model and is_data_provided:
        message = f"Use the {preferred_model} model. {request.message}"

    if is_data_provided:
        try:
            agent_response = agent.run(request.message)
            numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", agent_response)
            if len(numbers) >= 4:
                quarterly_growth = [float(n) for n in numbers[:4]]
                r2_score = None
                r2_match = re.search(r"r2[\s:=-]*([0-9]*\.?[0-9]+)", agent_response, re.IGNORECASE)
                if r2_match:
                    r2_score = float(r2_match.group(1))
                analysis_prompt = (
                    "Here are the results of the financial forecast:\n"
                    f"Predicted ARR YoY growth for the next 4 quarters: {quarterly_growth}\n"
                    f"Model R² score: {r2_score if r2_score is not None else 'N/A'}\n"
                    "Please provide a clear, human-friendly analysis of these results, including any risks or opportunities you see."
                )
                analysis = llm.invoke(analysis_prompt).content
                return {
                    "response": analysis,
                    "data": {
                        "quarterly_growth": quarterly_growth,
                        "r2_score": r2_score
                    }
                }
            else:
                return {"response": agent_response}
        except Exception as e:
            return {"response": f"Sorry, there was an error: {str(e)}"}
    else:
        greeting = f"Hi {name}!" if name else "Hi!"
        project_info = GPT_INFO.get("project_info", {})
        project_info_str = " ".join([
            f"Creator: {project_info.get('creator', '')}.",
            f"Project name: {project_info.get('project_name', '')}.",
            f"Purpose: {project_info.get('purpose', '')}.",
            f"Contact: {project_info.get('contact', '')}."
        ])
        # --- GPRH summary for system prompt ---
        gprh = gprh_trend_analysis()
        gprh_summary = (
            f"Current GPRH index (geopolitical risk, last 12 months): {gprh['last_12_months_gprh']}. "
            f"Start: {gprh['start_value']}, End: {gprh['end_value']}, Change: {gprh['change']}. "
            f"Traffic light: {gprh['traffic_light']}. {gprh['opinion']} "
            "If the user asks about macroeconomic or geopolitical risk, use this information."
        )
        system_prompt = (
            f"{greeting} I'm your venture prediction assistant. "
            f"Project info: {project_info_str} "
            f"{gprh_summary} "
            "If the user asks about the project, answer using this info. "
            "If the user seems interested in predictions, you may offer, but don’t be pushy. "
            "Otherwise, just chat naturally. "
            "If you want a prediction, just tell me your data or upload a CSV!"
        )
        conversation = [
            {"role": "system", "content": system_prompt}
        ]
        for msg in history:
            if "role" in msg and "content" in msg:
                conversation.append({"role": msg["role"], "content": msg["content"]})
        conversation.append({"role": "user", "content": request.message})
        response = llm.invoke(conversation).content
        return {"response": response}

@app.get("/makro-analysis")
def makro_analysis():
    """Return the GPRH trend analysis for the last year."""
    gprh = gprh_trend_analysis()
    return {"gprh": gprh}

@app.get("/")
def root():
    """Root endpoint with API usage info."""
    return {"message": "ARR Growth Prediction API. Use /predict_raw, /predict_csv, or /chat."} 