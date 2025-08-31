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
    # Q1 2024
    q1_arr: float
    q1_net_new_arr: float
    q1_qrr: float
    q1_headcount: int
    q1_gross_margin: float
    q1_net_profit_loss: float
    # Q2 2024
    q2_arr: float
    q2_net_new_arr: float
    q2_qrr: float
    q2_headcount: int
    q2_gross_margin: float
    q2_net_profit_loss: float
    # Q3 2024
    q3_arr: float
    q3_net_new_arr: float
    q3_qrr: float
    q3_headcount: int
    q3_gross_margin: float
    q3_net_profit_loss: float
    # Q4 2024
    q4_arr: float
    q4_net_new_arr: float
    q4_qrr: float
    q4_headcount: int
    q4_gross_margin: float
    q4_net_profit_loss: float
    # Optional advanced mode
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
            "If the user seems interested in predictions, you may offer, but don't be pushy. "
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

@app.post("/predict")
def predict(request: GuidedInputRequest):
    """Unified forecasting endpoint that handles both basic and advanced mode using 4 quarters of data."""
    try:
        from guided_input_system import GuidedInputSystem
        from enhanced_prediction import EnhancedFinancialPredictor
        import pandas as pd
        
        # Initialize guided system
        guided_system = GuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Create DataFrame with 4 quarters of data
        quarters_data = []
        
        # Q1 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q1',
            'Quarter Num': 1,
            'cARR': request.q1_arr,
            'Net New ARR': request.q1_net_new_arr,
            'QRR': request.q1_qrr,
            'Headcount (HC)': request.q1_headcount,
            'Gross Margin (in %)': request.q1_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q1_net_profit_loss
        })
        
        # Q2 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q2',
            'Quarter Num': 2,
            'cARR': request.q2_arr,
            'Net New ARR': request.q2_net_new_arr,
            'QRR': request.q2_qrr,
            'Headcount (HC)': request.q2_headcount,
            'Gross Margin (in %)': request.q2_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q2_net_profit_loss
        })
        
        # Q3 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q3',
            'Quarter Num': 3,
            'cARR': request.q3_arr,
            'Net New ARR': request.q3_net_new_arr,
            'QRR': request.q3_qrr,
            'Headcount (HC)': request.q3_headcount,
            'Gross Margin (in %)': request.q3_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q3_net_profit_loss
        })
        
        # Q4 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q4',
            'Quarter Num': 4,
            'cARR': request.q4_arr,
            'Net New ARR': request.q4_net_new_arr,
            'QRR': request.q4_qrr,
            'Headcount (HC)': request.q4_headcount,
            'Gross Margin (in %)': request.q4_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q4_net_profit_loss
        })
        
        # Create DataFrame
        forecast_df = pd.DataFrame(quarters_data)
        
        # Calculate additional required fields for LightGBM model (same as CSV processing)
        for i, row in forecast_df.iterrows():
            # Calculate ARR YoY Growth
            if i == 0:  # Q1 - use Q1 data as baseline
                forecast_df.loc[i, 'ARR YoY Growth (in %)'] = 0
            else:
                prev_arr = forecast_df.loc[i-1, 'cARR']
                if prev_arr > 0:
                    forecast_df.loc[i, 'ARR YoY Growth (in %)'] = ((row['cARR'] - prev_arr) / prev_arr) * 100
                else:
                    forecast_df.loc[i, 'ARR YoY Growth (in %)'] = 0
            
            # Calculate Revenue YoY Growth (same as ARR for SaaS)
            forecast_df.loc[i, 'Revenue YoY Growth (in %)'] = forecast_df.loc[i, 'ARR YoY Growth (in %)']
            
            # Calculate Magic Number (Sales Efficiency)
            if row['Net New ARR'] > 0:
                # Estimate Sales & Marketing based on typical Magic Number
                typical_magic_number = 0.7
                estimated_sales_marketing = row['Net New ARR'] / typical_magic_number
                forecast_df.loc[i, 'Sales & Marketing'] = min(estimated_sales_marketing, row['cARR'] * 0.8)
            else:
                forecast_df.loc[i, 'Sales & Marketing'] = 0
            
            # Calculate EBITDA (simplified)
            forecast_df.loc[i, 'EBITDA'] = row['cARR'] * (row['Gross Margin (in %)'] / 100) - row['Sales & Marketing']
            
            # Calculate Cash Burn (simplified)
            forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] = -forecast_df.loc[i, 'EBITDA']
            
            # Calculate LTM Rule of 40% (ARR)
            forecast_df.loc[i, 'LTM Rule of 40% (ARR)'] = row['ARR YoY Growth (in %)'] + (row['Gross Margin (in %)'] - 50)
            
            # Calculate Magic Number
            if forecast_df.loc[i, 'Sales & Marketing'] > 0:
                forecast_df.loc[i, 'Magic_Number'] = row['Net New ARR'] / forecast_df.loc[i, 'Sales & Marketing']
            else:
                forecast_df.loc[i, 'Magic_Number'] = 0
            
            # Calculate Burn Multiple
            if forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] != 0:
                forecast_df.loc[i, 'Burn_Multiple'] = abs(forecast_df.loc[i, 'Net New ARR'] / forecast_df.loc[i, 'Cash Burn (OCF & ICF)'])
            else:
                forecast_df.loc[i, 'Burn_Multiple'] = 0
            
            # Calculate ARR per Headcount
            if row['Headcount (HC)'] > 0:
                forecast_df.loc[i, 'ARR_per_Headcount'] = row['cARR'] / row['Headcount (HC)']
            else:
                forecast_df.loc[i, 'ARR_per_Headcount'] = 0
        
        # Add missing columns required by LightGBM model (same as CSV processing)
        if 'Sales & Marketing' not in forecast_df.columns:
            # Estimate Sales & Marketing based on Net New ARR and typical Magic Number
            typical_magic_number = 0.7
            forecast_df['Sales & Marketing'] = forecast_df['Net New ARR'] / typical_magic_number
            # Cap at reasonable levels (not more than 80% of ARR)
            forecast_df['Sales & Marketing'] = forecast_df['Sales & Marketing'].clip(upper=forecast_df['cARR'] * 0.8)
        
        # Ensure all required fields are present (same as CSV processing)
        required_fields = [
            "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
            "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
        ]
        
        for field in required_fields:
            if field not in forecast_df.columns:
                if field == 'EBITDA':
                    forecast_df[field] = forecast_df['cARR'] * 0.2  # Estimate 20% of ARR
                elif field == 'Cash Burn (OCF & ICF)':
                    forecast_df[field] = -forecast_df['cARR'] * 0.3  # Estimate -30% of ARR
                elif field == 'LTM Rule of 40% (ARR)':
                    forecast_df[field] = forecast_df['ARR YoY Growth (in %)'] + forecast_df['Gross Margin (in %)'] * 0.2
                elif field == 'Quarter Num':
                    forecast_df[field] = range(1, len(forecast_df) + 1)
                else:
                    forecast_df[field] = 0
        
        # Add any additional fields that might be expected by the LightGBM model
        if 'Net_Profit_Loss_Margin_Percent' not in forecast_df.columns:
            forecast_df['Net_Profit_Loss_Margin_Percent'] = forecast_df['Net Profit/Loss Margin (in %)']
        
        # Handle advanced mode if enabled - smart override logic
        if request.advanced_mode and request.advanced_metrics:
            print(f"🔧 Advanced mode enabled - smart override logic")
            
            # Apply advanced metrics to each quarter
            for i, row in forecast_df.iterrows():
                quarter_num = i + 1
                quarter_key = f"q{quarter_num}"
                
                # Apply quarter-specific advanced metrics if available
                if quarter_key in request.advanced_metrics:
                    quarter_metrics = request.advanced_metrics[quarter_key]
                    
                    # Smart override logic: 0 = estimate automatically, >0 = use user value
                    for metric_name, metric_value in quarter_metrics.items():
                        if metric_value == 0:
                            # User set to 0 - estimate automatically (keep current estimate)
                            print(f"  🔧 Q{quarter_num} {metric_name}: 0 → estimating automatically")
                            continue
                        elif metric_value > 0 or metric_value < 0:  # Allow negative values for cash burn, churn, etc.
                            # User provided a value - override the estimate
                            if metric_name == 'headcount':
                                forecast_df.loc[i, 'Headcount (HC)'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'sales_marketing':
                                forecast_df.loc[i, 'Sales & Marketing'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'cash_burn':
                                forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'gross_margin':
                                forecast_df.loc[i, 'Gross Margin (in %)'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}%")
                            elif metric_name == 'customers_eop':
                                forecast_df.loc[i, 'Customers (EoP)'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'expansion_upsell':
                                forecast_df.loc[i, 'Expansion & Upsell'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'churn_reduction':
                                forecast_df.loc[i, 'Churn & Reduction'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'ebitda':
                                forecast_df.loc[i, 'EBITDA'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            elif metric_name == 'ltm_rule_40':
                                forecast_df.loc[i, 'LTM Rule of 40% (ARR)'] = metric_value
                                print(f"  🔧 Q{quarter_num} {metric_name}: overrode with {metric_value}")
                            else:
                                print(f"  ⚠️ Q{quarter_num} {metric_name}: unknown metric, skipping")
                
                # Apply global overrides if available (affects all quarters)
                if 'global' in request.advanced_metrics:
                    global_metrics = request.advanced_metrics['global']
                    for metric_name, metric_value in global_metrics.items():
                        if metric_value == 0:
                            # User set to 0 - estimate automatically
                            print(f"  🔧 Global {metric_name}: 0 → estimating automatically")
                            continue
                        elif metric_name == 'magic_number_override':
                            # Override Magic Number calculation for all quarters
                            for j in range(len(forecast_df)):
                                if forecast_df.loc[j, 'Sales & Marketing'] > 0:
                                    forecast_df.loc[j, 'Magic_Number'] = forecast_df.loc[j, 'Net New ARR'] / metric_value
                            print(f"  🔧 Applied global Magic Number override: {metric_value}")
                        elif metric_name == 'burn_multiple_override':
                            # Override Burn Multiple calculation for all quarters
                            for j in range(len(forecast_df)):
                                if forecast_df.loc[j, 'Cash Burn (OCF & ICF)'] != 0:
                                    forecast_df.loc[j, 'Burn_Multiple'] = abs(forecast_df.loc[j, 'Net New ARR'] / metric_value)
                            print(f"  🔧 Applied global Burn Multiple override: {metric_value}")
                        elif metric_name == 'arr_per_headcount_override':
                            # Override ARR per Headcount for all quarters
                            for j in range(len(forecast_df)):
                                if forecast_df.loc[j, 'Headcount (HC)'] > 0:
                                    forecast_df.loc[j, 'ARR_per_Headcount'] = metric_value
                            print(f"  🔧 Applied global ARR per Headcount override: {metric_value}")
                        else:
                            print(f"  ⚠️ Global {metric_name}: unknown metric, skipping")
        
        # Ensure the DataFrame has the same structure as CSV processing
        print(f"🔍 Guided forecast DataFrame shape: {forecast_df.shape}")
        print(f"🔍 Guided forecast DataFrame columns: {list(forecast_df.columns)}")
        print(f"🔍 First row sample: {forecast_df.iloc[0].to_dict()}")
        
        # Try to make prediction with trained model
        try:
            from financial_prediction import load_trained_model, predict_future_arr
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                forecast_results = predict_future_arr(trained_model, forecast_df)
                model_used = "LightGBM Model"
                forecast_success = True
            else:
                raise Exception("No trained model available")
        except Exception as e:
            # Use fallback calculation
            predictor = EnhancedFinancialPredictor()
            # Use all 4 quarters for fallback calculation instead of just the latest
            print(f"⚠️ LightGBM failed, using fallback calculation with 4 quarters of data")
            try:
                # Try to use all 4 quarters for better fallback prediction
                forecast_results = predictor._generate_fallback_forecast(forecast_df.to_dict('records'))
                model_used = "Fallback Calculation (4 Quarters)"
                forecast_success = False
            except Exception as fallback_error:
                # If that fails, fall back to single quarter
                print(f"⚠️ Multi-quarter fallback failed, using single quarter: {fallback_error}")
                latest_quarter = forecast_df.iloc[-1].to_dict()
                forecast_results = predictor._generate_fallback_forecast(latest_quarter)
                model_used = "Fallback Calculation (Single Quarter)"
                forecast_success = False
        
        # Generate insights based on latest quarter
        latest_q4 = forecast_df.iloc[-1]
        insights = {
            'size_category': 'Early Stage' if latest_q4['cARR'] < 1e6 else 'Growth Stage' if latest_q4['cARR'] < 10e6 else 'Scale Stage' if latest_q4['cARR'] < 100e6 else 'Enterprise',
            'growth_insight': f"Q4 Growth rate: {latest_q4['ARR YoY Growth (in %)']:.1f}%",
            'efficiency_insight': f"Q4 Magic Number: {latest_q4['Magic_Number']:.2f}",
            'quarterly_trend': f"ARR growth: Q1: {forecast_df.iloc[0]['cARR']:,.0f} → Q4: {latest_q4['cARR']:,.0f}",
            'headcount_trend': f"Team growth: Q1: {forecast_df.iloc[0]['Headcount (HC)']} → Q4: {latest_q4['Headcount (HC)']}"
        }
        
        return {
            "company_name": request.company_name,
            "input_metrics": forecast_df.to_dict('records'),
            "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
            "insights": insights,
            "model_used": model_used,
            "forecast_success": forecast_success,
            "message": "Guided forecast completed successfully using 4 quarters of data!"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Guided forecast failed: {str(e)}"}
        )

@app.post("/advanced_forecast")
def advanced_forecast(request: GuidedInputRequest):
    """Advanced forecasting endpoint that handles comprehensive financial data including advanced metrics."""
    try:
        from guided_input_system import GuidedInputSystem
        from enhanced_prediction import EnhancedFinancialPredictor
        import pandas as pd
        
        # Initialize guided system
        guided_system = GuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Create DataFrame with 4 quarters of data (same as guided_forecast)
        quarters_data = []
        
        # Q1 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q1',
            'Quarter Num': 1,
            'cARR': request.q1_arr,
            'Net New ARR': request.q1_net_new_arr,
            'QRR': request.q1_qrr,
            'Headcount (HC)': request.q1_headcount,
            'Gross Margin (in %)': request.q1_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q1_net_profit_loss
        })
        
        # Q2 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q2',
            'Quarter Num': 2,
            'cARR': request.q2_arr,
            'Net New ARR': request.q2_net_new_arr,
            'QRR': request.q2_qrr,
            'Headcount (HC)': request.q2_headcount,
            'Gross Margin (in %)': request.q2_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q2_net_profit_loss
        })
        
        # Q3 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q3',
            'Quarter Num': 3,
            'cARR': request.q3_arr,
            'Net New ARR': request.q3_net_new_arr,
            'QRR': request.q3_qrr,
            'Headcount (HC)': request.q3_headcount,
            'Gross Margin (in %)': request.q3_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q3_net_profit_loss
        })
        
        # Q4 2024
        quarters_data.append({
            'id_company': request.company_name,
            'Financial Quarter': 'FY24 Q4',
            'Quarter Num': 4,
            'cARR': request.q4_arr,
            'Net New ARR': request.q4_net_new_arr,
            'QRR': request.q4_qrr,
            'Headcount (HC)': request.q4_headcount,
            'Gross Margin (in %)': request.q4_gross_margin,
            'Net Profit/Loss Margin (in %)': request.q4_net_profit_loss
        })
        
        # Create DataFrame
        forecast_df = pd.DataFrame(quarters_data)
        
        # Apply all the same calculations as guided_forecast
        # Calculate additional required fields for LightGBM model (same as CSV processing)
        for i, row in forecast_df.iterrows():
            # Calculate ARR YoY Growth
            if i == 0:  # Q1 - use Q1 data as baseline
                forecast_df.loc[i, 'ARR YoY Growth (in %)'] = 0
            else:
                prev_arr = forecast_df.loc[i-1, 'cARR']
                if prev_arr > 0:
                    forecast_df.loc[i, 'ARR YoY Growth (in %)'] = ((row['cARR'] - prev_arr) / prev_arr) * 100
                else:
                    forecast_df.loc[i, 'ARR YoY Growth (in %)'] = 0
            
            # Calculate Revenue YoY Growth (same as ARR for SaaS)
            forecast_df.loc[i, 'Revenue YoY Growth (in %)'] = forecast_df.loc[i, 'ARR YoY Growth (in %)']
            
            # Calculate Magic Number (Sales Efficiency)
            if row['Net New ARR'] > 0:
                # Estimate Sales & Marketing based on typical Magic Number
                typical_magic_number = 0.7
                estimated_sales_marketing = row['Net New ARR'] / typical_magic_number
                forecast_df.loc[i, 'Sales & Marketing'] = min(estimated_sales_marketing, row['cARR'] * 0.8)
            else:
                forecast_df.loc[i, 'Sales & Marketing'] = 0
            
            # Calculate EBITDA (simplified)
            forecast_df.loc[i, 'EBITDA'] = row['cARR'] * (row['Gross Margin (in %)'] / 100) - row['Sales & Marketing']
            
            # Calculate Cash Burn (simplified)
            forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] = -forecast_df.loc[i, 'EBITDA']
            
            # Calculate LTM Rule of 40% (ARR)
            forecast_df.loc[i, 'LTM Rule of 40% (ARR)'] = row['ARR YoY Growth (in %)'] + (row['Gross Margin (in %)'] - 50)
            
            # Calculate Magic Number
            if forecast_df.loc[i, 'Sales & Marketing'] > 0:
                forecast_df.loc[i, 'Magic_Number'] = row['Net New ARR'] / forecast_df.loc[i, 'Sales & Marketing']
            else:
                forecast_df.loc[i, 'Magic_Number'] = 0
            
            # Calculate Burn Multiple
            if forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] != 0:
                forecast_df.loc[i, 'Burn_Multiple'] = abs(forecast_df.loc[i, 'Net New ARR'] / forecast_df.loc[i, 'Cash Burn (OCF & ICF)'])
            else:
                forecast_df.loc[i, 'Burn_Multiple'] = 0
            
            # Calculate ARR per Headcount
            if row['Headcount (HC)'] > 0:
                forecast_df.loc[i, 'ARR_per_Headcount'] = row['cARR'] / row['Headcount (HC)']
            else:
                forecast_df.loc[i, 'ARR_per_Headcount'] = 0
        
        # Add missing columns required by LightGBM model (same as CSV processing)
        if 'Sales & Marketing' not in forecast_df.columns:
            # Estimate Sales & Marketing based on Net New ARR and typical Magic Number
            typical_magic_number = 0.7
            forecast_df['Sales & Marketing'] = forecast_df['Net New ARR'] / typical_magic_number
            # Cap at reasonable levels (not more than 80% of ARR)
            forecast_df['Sales & Marketing'] = forecast_df['Sales & Marketing'].clip(upper=forecast_df['cARR'] * 0.8)
        
        # Ensure all required fields are present (same as CSV processing)
        required_fields = [
            "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
            "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
        ]
        
        for field in required_fields:
            if field not in forecast_df.columns:
                if field == 'EBITDA':
                    forecast_df[field] = forecast_df['cARR'] * 0.2  # Estimate 20% of ARR
                elif field == 'Cash Burn (OCF & ICF)':
                    forecast_df[field] = forecast_df['cARR'] * 0.3  # Estimate -30% of ARR
                elif field == 'LTM Rule of 40% (ARR)':
                    forecast_df[field] = forecast_df['ARR YoY Growth (in %)'] + forecast_df['Gross Margin (in %)'] * 0.2
                elif field == 'Quarter Num':
                    forecast_df[field] = range(1, len(forecast_df) + 1)
                else:
                    forecast_df[field] = 0
        
        # Add any additional fields that might be expected by the LightGBM model
        if 'Net_Profit_Loss_Margin_Percent' not in forecast_df.columns:
            forecast_df['Net_Profit_Loss_Margin_Percent'] = forecast_df['Net Profit/Loss Margin (in %)']
        
        # Enhanced advanced mode handling - override estimated metrics for better accuracy
        if request.advanced_mode and request.advanced_metrics:
            print(f"🔧 Advanced mode enabled - overriding estimated metrics for better accuracy")
            
            # Apply advanced metrics to each quarter
            for i, row in forecast_df.iterrows():
                quarter_num = i + 1
                quarter_key = f"q{quarter_num}"
                
                # Apply quarter-specific advanced metrics if available
                if quarter_key in request.advanced_metrics:
                    quarter_metrics = request.advanced_metrics[quarter_key]
                    
                    # Override only the estimated/inferred metrics that the system calculates
                    for metric_name, metric_value in quarter_metrics.items():
                        if metric_name == 'sales_marketing':
                            # Override the estimated Sales & Marketing
                            forecast_df.loc[i, 'Sales & Marketing'] = metric_value
                            print(f"  🔧 Overrode Sales & Marketing: {metric_value} for Q{quarter_num}")
                        elif metric_name == 'cash_burn':
                            # Override the estimated Cash Burn
                            forecast_df.loc[i, 'Cash Burn (OCF & ICF)'] = metric_value
                            print(f"  🔧 Overrode Cash Burn: {metric_value} for Q{quarter_num}")
                        elif metric_name == 'ebitda':
                            # Override the estimated EBITDA
                            forecast_df.loc[i, 'EBITDA'] = metric_value
                            print(f"  🔧 Overrode EBITDA: {metric_value} for Q{quarter_num}")
                        elif metric_name == 'customers_eop':
                            # Override the estimated Customers (EoP)
                            forecast_df.loc[i, 'Customers (EoP)'] = metric_value
                            print(f"  🔧 Overrode Customers (EoP): {metric_value} for Q{quarter_num}")
                        elif metric_name == 'expansion_upsell':
                            # Override the estimated Expansion & Upsell
                            forecast_df.loc[i, 'Expansion & Upsell'] = metric_value
                            print(f"  🔧 Overrode Expansion & Upsell: {metric_value} for Q{quarter_num}")
                        elif metric_name == 'churn_reduction':
                            # Override the estimated Churn & Reduction
                            forecast_df.loc[i, 'Churn & Reduction'] = metric_value
                            print(f"  🔧 Overrode Churn & Reduction: {metric_value} for Q{quarter_num}")
                        elif metric_name == 'gross_margin':
                            # Override the estimated Gross Margin
                            forecast_df.loc[i, 'Gross Margin (in %)'] = metric_value
                            print(f"  🔧 Overrode Gross Margin: {metric_value}% for Q{quarter_num}")
                
                # Apply global overrides if available (affects all quarters)
                if 'global' in request.advanced_metrics:
                    global_metrics = request.advanced_metrics['global']
                    for metric_name, metric_value in global_metrics.items():
                        if metric_name == 'magic_number_override':
                            # Override Magic Number calculation for all quarters
                            for j in range(len(forecast_df)):
                                if forecast_df.loc[j, 'Sales & Marketing'] > 0:
                                    forecast_df.loc[j, 'Magic_Number'] = forecast_df.loc[j, 'Net New ARR'] / metric_value
                            print(f"  🔧 Applied global Magic Number override: {metric_value}")
                        elif metric_name == 'burn_multiple_override':
                            # Override Burn Multiple calculation for all quarters
                            for j in range(len(forecast_df)):
                                if forecast_df.loc[j, 'Cash Burn (OCF & ICF)'] != 0:
                                    forecast_df.loc[j, 'Burn_Multiple'] = abs(forecast_df.loc[j, 'Net New ARR'] / metric_value)
                            print(f"  🔧 Applied global Burn Multiple override: {metric_value}")
                        elif metric_name == 'ltm_rule_40_override':
                            # Override LTM Rule of 40% calculation for all quarters
                            for j in range(len(forecast_df)):
                                forecast_df.loc[j, 'LTM Rule of 40% (ARR)'] = metric_value
                            print(f"  🔧 Applied global LTM Rule of 40% override: {metric_value}")
        
        # Ensure the DataFrame has the same structure as CSV processing
        print(f"🔍 Advanced forecast DataFrame shape: {forecast_df.shape}")
        print(f"🔍 Advanced forecast DataFrame columns: {list(forecast_df.columns)}")
        print(f"🔍 First row sample: {forecast_df.iloc[0].to_dict()}")
        
        # Try to make prediction with trained model
        try:
            from financial_prediction import load_trained_model, predict_future_arr
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                forecast_results = predict_future_arr(trained_model, forecast_df)
                model_used = "LightGBM Model (Advanced)"
                forecast_success = True
            else:
                raise Exception("No trained model available")
        except Exception as e:
            # Use fallback calculation
            predictor = EnhancedFinancialPredictor()
            # Use all 4 quarters for fallback calculation instead of just the latest
            print(f"⚠️ LightGBM failed, using fallback calculation with 4 quarters of data")
            try:
                # Try to use all 4 quarters for better fallback prediction
                forecast_results = predictor._generate_fallback_forecast(forecast_df.to_dict('records'))
                model_used = "Fallback Calculation (4 Quarters, Advanced)"
                forecast_success = False
            except Exception as fallback_error:
                # If that fails, fall back to single quarter
                print(f"⚠️ Multi-quarter fallback failed, using single quarter: {fallback_error}")
                latest_quarter = forecast_df.iloc[-1].to_dict()
                forecast_results = predictor._generate_fallback_forecast(latest_quarter)
                model_used = "Fallback Calculation (Single Quarter, Advanced)"
                forecast_success = False
        
        # Generate enhanced insights based on all quarters and advanced metrics
        latest_q4 = forecast_df.iloc[-1]
        insights = {
            'size_category': 'Early Stage' if latest_q4['cARR'] < 1e6 else 'Growth Stage' if latest_q4['cARR'] < 10e6 else 'Scale Stage' if latest_q4['cARR'] < 100e6 else 'Enterprise',
            'growth_insight': f"Q4 Growth rate: {latest_q4['ARR YoY Growth (in %)']:.1f}%",
            'efficiency_insight': f"Q4 Magic Number: {latest_q4['Magic_Number']:.2f}",
            'quarterly_trend': f"ARR growth: Q1: {forecast_df.iloc[0]['cARR']:,.0f} → Q4: {latest_q4['cARR']:,.0f}",
            'headcount_trend': f"Team growth: Q1: {forecast_df.iloc[0]['Headcount (HC)']} → Q4: {latest_q4['Headcount (HC)']}",
            'advanced_metrics_used': request.advanced_mode and request.advanced_metrics is not None,
            'total_advanced_metrics': len(request.advanced_metrics) if request.advanced_mode and request.advanced_metrics else 0
        }
        
        # Add advanced insights if available
        if request.advanced_mode and request.advanced_metrics:
            insights['advanced_mode_info'] = f"Advanced mode overrode {request.advanced_metrics_count} estimated metrics for better accuracy"
            if 'Customers (EoP)' in forecast_df.columns:
                insights['customer_growth'] = f"Customer growth: Q1: {forecast_df.iloc[0].get('Customers (EoP)', 0):,.0f} → Q4: {latest_q4.get('Customers (EoP)', 0):,.0f}"
            if 'Sales & Marketing' in forecast_df.columns:
                insights['sales_efficiency'] = f"Sales & Marketing efficiency: Q4 Magic Number: {latest_q4.get('Magic_Number', 0):.2f}"
            if 'Cash Burn (OCF & ICF)' in forecast_df.columns:
                insights['cash_burn_insight'] = f"Cash burn pattern: Q4 Burn Multiple: {latest_q4.get('Burn_Multiple', 0):.2f}"
        
        return {
            "company_name": request.company_name,
            "input_metrics": forecast_df.to_dict('records'),
            "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
            "insights": insights,
            "model_used": model_used,
            "forecast_success": forecast_success,
            "advanced_mode_enabled": request.advanced_mode,
            "advanced_metrics_count": len(request.advanced_metrics) if request.advanced_metrics else 0,
            "message": "Advanced forecast completed successfully using 4 quarters of data with comprehensive metrics!"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Advanced forecast failed: {str(e)}"}
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
            "predict": "POST /predict - NEW: Unified forecasting endpoint for both basic and advanced mode (requires 4 quarters of data)",
            "makro_analysis": "GET /makro_analysis - Macroeconomic indicators"
        },
        "new_feature": "Unified Forecasting System - Single endpoint that handles both basic guided input and advanced metrics override for sophisticated financial modeling!"
    } 