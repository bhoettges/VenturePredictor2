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
from gpr_analysis import gprh_trend_analysis
from vix_analysis import vix_trend_analysis
from move_analysis import move_trend_analysis

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

# Initialize model variables
model_xgb = None
model_rf = None

print("ℹ️  Only LightGBM model is available. XGBoost and Random Forest models not found.")

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

class GuidedInputRequest(BaseModel):
    company_name: str
    current_arr: float
    net_new_arr: float
    growth_rate: float = None
    advanced_mode: bool = False
    advanced_metrics: dict = None

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
def predict_csv(file: UploadFile = File(...), model: str = Form('lightgbm')):
    """Predict ARR growth from a CSV file upload using LightGBM model."""
    try:
        df = pd.read_csv(file.file)
        if len(df) < 4:
            return JSONResponse(status_code=400, content={"error": "CSV must have at least 4 rows (quarters)."})
        
        # Handle your CSV format (ARR_End_of_Quarter, Quarterly_Net_New_ARR, etc.)
        # Convert to the format expected by the LightGBM model
        df_processed = df.copy()
        
        # Rename columns to match expected format
        column_mapping = {
            'ARR_End_of_Quarter': 'cARR',
            'Quarterly_Net_New_ARR': 'Net New ARR',
            'QRR_Quarterly_Recurring_Revenue': 'QRR',
            'Headcount': 'Headcount (HC)',
            'Gross_Margin_Percent': 'Gross Margin (in %)',
            'Net_Profit_Loss_Margin_Percent': 'Net_Profit_Loss_Margin_Percent'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed[new_col] = df_processed[old_col]
        
        # Add required fields that might be missing
        if 'ARR YoY Growth (in %)' not in df_processed.columns:
            df_processed['ARR YoY Growth (in %)'] = df_processed['cARR'].pct_change() * 100
        
        if 'Revenue YoY Growth (in %)' not in df_processed.columns:
            df_processed['Revenue YoY Growth (in %)'] = df_processed['QRR'].pct_change() * 100
        
        # Fill missing required fields with defaults
        required_fields = [
            "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
            "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
        ]
        
        for field in required_fields:
            if field not in df_processed.columns:
                if field == 'EBITDA':
                    df_processed[field] = df_processed['cARR'] * 0.2  # Estimate 20% of ARR
                elif field == 'Cash Burn (OCF & ICF)':
                    df_processed[field] = -df_processed['cARR'] * 0.3  # Estimate -30% of ARR
                elif field == 'LTM Rule of 40% (ARR)':
                    df_processed[field] = df_processed['ARR YoY Growth (in %)'] + df_processed['Gross Margin (in %)'] * 0.2
                elif field == 'Quarter Num':
                    df_processed[field] = range(1, len(df_processed) + 1)
                else:
                    df_processed[field] = 0
        
        # Add missing columns required by LightGBM model
        if 'Sales & Marketing' not in df_processed.columns:
            # Estimate Sales & Marketing based on Net New ARR and typical Magic Number
            # Magic Number = Net New ARR / Sales & Marketing
            # Typical Magic Number for SaaS companies: 0.5-1.0
            typical_magic_number = 0.7
            df_processed['Sales & Marketing'] = df_processed['Net New ARR'] / typical_magic_number
            # Cap at reasonable levels (not more than 80% of ARR)
            df_processed['Sales & Marketing'] = df_processed['Sales & Marketing'].clip(upper=df_processed['cARR'] * 0.8)
        
        # Get last 4 quarters
        df_last4 = df_processed.tail(4)
        
        # Extract features in the order expected by the model
        features = []
        for _, row in df_last4.iterrows():
            for field in REQUIRED_FIELDS:
                features.append(float(row[field]))
        
        if len(features) != 28:
            return JSONResponse(status_code=400, content={"error": f"Expected 28 features, got {len(features)}"})
        
        # Use LightGBM model for prediction
        try:
            from financial_prediction import load_trained_model, predict_future_arr
            
            # Load the LightGBM model
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                # Create a DataFrame with the processed data
                forecast_df = df_processed.copy()
                forecast_df['id_company'] = 'Uploaded Company'
                forecast_df['Financial Quarter'] = [f'FY24 Q{i}' for i in range(1, len(forecast_df) + 1)]
                
                # Make prediction using the LightGBM model
                print(f"🔍 Attempting LightGBM prediction with {len(forecast_df)} rows")
                print(f"🔍 Forecast DataFrame columns: {list(forecast_df.columns)}")
                print(f"🔍 First row sample: {forecast_df.iloc[0].to_dict()}")
                
                try:
                    forecast_results = predict_future_arr(trained_model, forecast_df)
                    print(f"✅ LightGBM prediction successful: {type(forecast_results)}")
                    model_used = 'LightGBM'
                except Exception as e:
                    print(f"❌ LightGBM prediction failed: {str(e)}")
                    raise e
                
                return {
                    "model": model_used,
                    "prediction": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
                    "message": "LightGBM forecast completed successfully!"
                }
            else:
                raise Exception("LightGBM model not available")
                
        except Exception as e:
            # Fallback to simple calculation if LightGBM fails and no other models available
            if model_xgb is None and model_rf is None:
                # Simple growth projection as fallback
                last_arr = df_processed['cARR'].iloc[-1]
                growth_rate = df_processed['ARR YoY Growth (in %)'].iloc[-1] / 100
                
                # Project next 4 quarters with slight deceleration
                future_arr = []
                for i in range(1, 5):
                    deceleration = 0.95 ** i  # 5% deceleration per quarter
                    projected_growth = growth_rate * deceleration
                    future_arr.append(last_arr * (1 + projected_growth))
                
                return {
                    "model": "Fallback Calculation",
                    "prediction": [{"quarter": f"Q{i+1}", "projected_arr": arr} for i, arr in enumerate(future_arr)],
                    "message": "LightGBM failed, used fallback calculation"
                }
            else:
                # Fallback to existing models if available
                model_obj = model_rf if model == 'random_forest' else model_xgb
                if model_obj is not None:
                    prediction = model_obj.predict(np.array([features]))
                    model_used = 'Random Forest' if model == 'random_forest' else 'XGBoost'
                    return {
                        "model": model_used,
                        "prediction": prediction.tolist(),
                        "message": f"LightGBM failed, used {model_used} as fallback"
                    }
                else:
                    return JSONResponse(status_code=500, content={"error": "No models available for prediction"})
            
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
            # Extract numbers from the message
            numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", request.message)
            
            if len(numbers) >= 28:
                # User provided 28 features directly - use LightGBM
                print(f"🔍 Chat detected {len(numbers)} features, using LightGBM model")
                
                try:
                    from financial_prediction import load_trained_model, predict_future_arr
                    
                    # Load LightGBM model
                    trained_model = load_trained_model('lightgbm_financial_model.pkl')
                    if trained_model:
                        # Create a sample DataFrame with the features
                        # We need to create a proper DataFrame structure for the LightGBM model
                        # This is a simplified approach - in practice, you might want to map these to proper columns
                        
                        # For now, let's use the guided forecast approach with extracted data
                        from guided_input_system import GuidedInputSystem
                        
                        guided_system = GuidedInputSystem()
                        guided_system.initialize_from_training_data()
                        
                        # Try to extract meaningful data from the 28 features
                        # Assuming the features are in the order: [cARR, Net New ARR, QRR, Headcount, Gross Margin, Net Profit Loss, Quarter Num] x 4 quarters
                        if len(numbers) >= 28:
                            # Extract last quarter data (most recent)
                            last_quarter_features = numbers[-7:]  # Last 7 features
                            
                            # Create input for guided forecast
                            primary_inputs = {
                                'cARR': float(last_quarter_features[0]) if len(last_quarter_features) > 0 else 1000000,
                                'Net New ARR': float(last_quarter_features[1]) if len(last_quarter_features) > 1 else 200000,
                                'ARR YoY Growth (in %)': 15.0,  # Default if not calculable
                                'Quarter Num': 4
                            }
                            
                            # Infer secondary metrics
                            inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
                            
                            # Add required fields for the DataFrame
                            inferred_metrics['id_company'] = 'Chat User'
                            inferred_metrics['Financial Quarter'] = 'FY24 Q4'
                            
                            # Create forecast-ready DataFrame
                            forecast_df = guided_system.create_forecast_input(inferred_metrics)
                            
                            # Make prediction using LightGBM
                            forecast_results = predict_future_arr(trained_model, forecast_df)
                            
                            # Generate analysis
                            analysis_prompt = (
                                "Here are the results of the LightGBM financial forecast:\n"
                                f"Predicted ARR YoY growth for the next 4 quarters: {forecast_results['Predicted YoY Growth (%)'].tolist()}\n"
                                f"Predicted Absolute ARR values: {forecast_results['Predicted Absolute cARR (€)'].tolist()}\n"
                                "Please provide a clear, human-friendly analysis of these results, including any risks or opportunities you see."
                            )
                            analysis = llm.invoke(analysis_prompt).content
                            
                            return {
                                "response": analysis,
                                "data": {
                                    "model": "LightGBM",
                                    "forecast_results": forecast_results.to_dict('records'),
                                    "message": "LightGBM forecast completed successfully via chat!"
                                }
                            }
                        else:
                            return {"response": "I detected some numbers but need exactly 28 features for a proper forecast. Please provide complete financial data for 4 quarters."}
                    else:
                        return {"response": "Sorry, the LightGBM model is not available right now."}
                        
                except Exception as e:
                    print(f"❌ LightGBM prediction failed in chat: {str(e)}")
                    return {"response": f"Sorry, there was an error running the LightGBM forecast: {str(e)}"}
                    
            elif "csv" in message.lower():
                # User mentioned CSV - guide them to use the CSV endpoint
                return {
                    "response": "I can help you analyze CSV data! Please use the `/predict_csv` endpoint to upload your CSV file, or you can paste your financial data directly in the chat (I need 28 features for 4 quarters)."
                }
            else:
                # Not enough data - ask for more
                return {
                    "response": "I can help you with financial forecasting! Please provide your company's financial data. I need 28 features covering 4 quarters, or you can mention if you have a CSV file."
                }
                
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
        # --- GPRH, VIX, and MOVE summary for system prompt ---
        gprh = gprh_trend_analysis()
        vix = vix_trend_analysis()
        move = move_trend_analysis()
        gprh_summary = (
            f"Current GPRH index (geopolitical risk, last 12 months): {gprh['last_12_months_gprh']}. "
            f"Start: {gprh['start_value']}, End: {gprh['end_value']}, Change: {gprh['change']}. "
            f"Traffic light: {gprh['traffic_light']}. {gprh['opinion']} "
        )
        vix_summary = (
            f"Current VIX index (market volatility, last 12 months): {vix['last_12_months_vix']}. "
            f"Start: {vix['start_value']}, End: {vix['end_value']}, Change: {vix['change']}. "
            f"Traffic light: {vix['traffic_light']}. {vix['opinion']} "
        )
        move_summary = (
            f"Current MOVE index (bond market volatility, last 12 months): {move['last_12_months_move']}. "
            f"Start: {move['start_value']}, End: {move['end_value']}, Change: {move['change']}. "
            f"Traffic light: {move['traffic_light']}. {move['opinion']} "
        )
        system_prompt = (
            f"{greeting} I'm your venture prediction assistant powered by LightGBM machine learning. "
            f"Project info: {project_info_str} "
            f"{gprh_summary} {vix_summary} {move_summary} "
            "If the user asks about macroeconomic, geopolitical risk, market volatility, or bond market volatility, use this information. "
            "If the user asks about the project, answer using this info. "
            "If the user seems interested in predictions, you may offer, but don’t be pushy. "
            "Otherwise, just chat naturally. "
            "I can provide financial forecasts using our trained LightGBM model. If you want a prediction, just provide your financial data (I need 28 features for 4 quarters) or mention if you have a CSV file!"
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
    """Return the GPRH, VIX, and MOVE trend analysis for the last year."""
    gprh = gprh_trend_analysis()
    vix = vix_trend_analysis()
    move = move_trend_analysis()
    return {"gprh": gprh, "vix": vix, "move": move}

@app.post("/guided_forecast")
def guided_forecast(request: GuidedInputRequest):
    """NEW: Guided forecasting with intelligent defaults."""
    try:
        from guided_input_system import GuidedInputSystem
        from enhanced_prediction import EnhancedFinancialPredictor
        
        # Initialize guided system
        guided_system = GuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Build primary inputs
        primary_inputs = {
            'cARR': request.current_arr,
            'Net New ARR': request.net_new_arr,
            'ARR YoY Growth (in %)': request.growth_rate or (request.net_new_arr / request.current_arr * 100) if request.current_arr > 0 else 0,
            'Quarter Num': 1
        }
        
        # Infer secondary metrics
        inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
        
        # Apply advanced mode overrides if provided
        if request.advanced_mode and request.advanced_metrics:
            for key, value in request.advanced_metrics.items():
                if key in inferred_metrics:
                    inferred_metrics[key] = value
        
        # Add company name
        inferred_metrics['id_company'] = request.company_name
        inferred_metrics['Financial Quarter'] = 'FY24 Q1'
        
        # Create forecast-ready DataFrame
        forecast_df = guided_system.create_forecast_input(inferred_metrics)
        
        # Try to make prediction with trained model
        try:
            from financial_prediction import load_trained_model, predict_future_arr
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                forecast_results = predict_future_arr(trained_model, forecast_df)
                model_used = "Trained Model"
                forecast_success = True
            else:
                raise Exception("No trained model available")
        except Exception as e:
            # Use fallback calculation
            predictor = EnhancedFinancialPredictor()
            forecast_results = predictor._generate_fallback_forecast(inferred_metrics)
            model_used = "Fallback Calculation"
            forecast_success = False
        
        # Generate insights
        insights = {
            'size_category': 'Early Stage' if request.current_arr < 1e6 else 'Growth Stage' if request.current_arr < 10e6 else 'Scale Stage' if request.current_arr < 100e6 else 'Enterprise',
            'growth_insight': f"Growth rate: {primary_inputs['ARR YoY Growth (in %)']:.1f}%",
            'efficiency_insight': f"Magic Number: {inferred_metrics.get('Magic_Number', 0):.2f}"
        }
        
        return {
            "company_name": request.company_name,
            "input_metrics": inferred_metrics,
            "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
            "insights": insights,
            "model_used": model_used,
            "forecast_success": forecast_success,
            "message": "Guided forecast completed successfully!"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Guided forecast failed: {str(e)}"}
        )

@app.get("/")
def root():
    """Root endpoint with API usage info."""
    return {
        "message": "Enhanced Financial Forecasting API", 
        "endpoints": {
            "predict_raw": "POST /predict_raw - Predict from raw features",
            "predict_csv": "POST /predict_csv - Predict from CSV upload",
            "chat": "POST /chat - Conversational AI interface",
            "guided_forecast": "POST /guided_forecast - NEW: Guided input with intelligent defaults",
            "makro_analysis": "GET /makro_analysis - Macroeconomic indicators"
        },
        "new_feature": "Guided Input System with Intelligent Defaults - Ask for only critical inputs and intelligently infer the rest!"
    } 