from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
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
from enhanced_guided_input import EnhancedGuidedInputSystem

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
print("ℹ️  Using LightGBM model for financial forecasting.")

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
    model: str = 'lightgbm'

class ChatRequest(BaseModel):
    message: str
    name: str = None
    preferred_model: str = None
    history: list = None

class AdvancedMetrics(BaseModel):
    """Advanced metrics that users can override in advanced mode."""
    sales_marketing: float = None
    ebitda: float = None
    cash_burn: float = None
    rule_of_40: float = None
    arr_yoy_growth: float = None
    revenue_yoy_growth: float = None
    magic_number: float = None
    burn_multiple: float = None
    customers_eop: float = None
    expansion_upsell: float = None
    churn_reduction: float = None
    gross_margin: float = None
    headcount: float = None
    net_profit_margin: float = None

class HistoricalARR(BaseModel):
    q1_arr: float = None  # 4 quarters ago
    q2_arr: float = None  # 3 quarters ago  
    q3_arr: float = None  # 2 quarters ago
    q4_arr: float = None  # 1 quarter ago (most recent)

class GuidedInputRequest(BaseModel):
    company_name: str
    current_arr: float
    net_new_arr: float
    historical_arr: HistoricalARR = None  # Optional historical ARR data
    advanced_mode: bool = False
    advanced_metrics: AdvancedMetrics = None

# Validation constants for enhanced mode
VALID_SECTORS = [
    "Cyber Security",
    "Data & Analytics", 
    "Infrastructure & Network",
    "Communication & Collaboration",
    "Marketing & Customer Experience",
    "Other"
]

VALID_COUNTRIES = [
    "United States",
    "Israel",
    "Germany", 
    "United Kingdom",
    "France",
    "Other"
]

VALID_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "Other"]

class EnhancedGuidedInputRequest(BaseModel):
    # Basic inputs (required)
    company_name: str = None
    current_arr: float
    net_new_arr: float
    
    # Enhanced mode (optional)
    enhanced_mode: bool = False
    
    # Enhanced inputs (optional, only used if enhanced_mode=True)
    sector: str = None
    country: str = None
    currency: str = None
    
    # Historical data (optional)
    historical_arr: HistoricalARR = None
    
    # Advanced metrics (optional)
    advanced_mode: bool = False
    advanced_metrics: AdvancedMetrics = None
    
    @validator('sector')
    def validate_sector(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_SECTORS:
                raise ValueError(f'Invalid sector. Must be one of: {", ".join(VALID_SECTORS)}')
        return v
    
    @validator('country')
    def validate_country(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_COUNTRIES:
                raise ValueError(f'Invalid country. Must be one of: {", ".join(VALID_COUNTRIES)}')
        return v
    
    @validator('currency')
    def validate_currency(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_CURRENCIES:
                raise ValueError(f'Invalid currency. Must be one of: {", ".join(VALID_CURRENCIES)}')
        return v

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
    # Use LightGBM model for all predictions
    from financial_prediction import load_trained_model, predict_future_arr
    try:
        model = load_trained_model('lightgbm_financial_model.pkl')
        if model:
            # Create a simple DataFrame for prediction
            import pandas as pd
            df = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(28)])
            result = predict_future_arr(model, df)
            return f"[LightGBM] Predicted ARR YoY growth for the next 4 quarters: {result}"
        else:
            return "Model not available. Please use the guided forecast endpoint instead."
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def csv_tool(input_str: str, model_choice: str = 'lightgbm'):
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
arr_growth_tool_lgbm = LC_Tool(
    name="LightGBM ARR Growth Predictor",
    func=lambda s: arr_tool(s, 'lightgbm'),
    description="Predicts ARR YoY growth for the next 4 quarters using the LightGBM model. Input: 28 comma-separated features."
)
csv_growth_tool_lgbm = LC_Tool(
    name="LightGBM CSV ARR Growth Predictor",
    func=lambda s: csv_tool(s, 'lightgbm'),
    description="Predicts ARR YoY growth for the next 4 quarters using LightGBM from a CSV file or CSV string."
)

# --- LangChain Agent ---
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = initialize_agent(
    tools=[arr_growth_tool_lgbm, csv_growth_tool_lgbm],
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

# --- API Endpoints ---
@app.post("/predict_raw")
def predict_raw(data: FeatureInput):
    """Predict ARR growth from raw features using LightGBM model."""
    if len(data.features) != 28:
        return JSONResponse(status_code=400, content={"error": "Exactly 28 features required (7 per quarter for 4 quarters)."})
    
    # Use LightGBM model for all predictions
    from financial_prediction import load_trained_model, predict_future_arr
    try:
        model = load_trained_model('lightgbm_financial_model.pkl')
        if model:
            import pandas as pd
            df = pd.DataFrame([data.features], columns=[f'feature_{i}' for i in range(28)])
            result = predict_future_arr(model, df)
            return {"model": "LightGBM", "prediction": result}
        else:
            return JSONResponse(status_code=500, content={"error": "LightGBM model not available"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...), model: str = Form('lightgbm')):
    """Predict ARR growth from a CSV file upload using the same logic as guided_forecast."""
    try:
        df = pd.read_csv(file.file)
        if len(df) < 1:
            return JSONResponse(status_code=400, content={"error": "CSV must have at least 1 row."})
        
        # Get the most recent row (last row)
        latest_row = df.iloc[-1]
        
        # Extract current ARR and Net New ARR from the CSV
        current_arr = None
        net_new_arr = None
        
        # Try different possible column names
        if 'ARR_End_of_Quarter' in latest_row:
            current_arr = latest_row['ARR_End_of_Quarter']
        elif 'cARR' in latest_row:
            current_arr = latest_row['cARR']
        elif 'ARR' in latest_row:
            current_arr = latest_row['ARR']
        
        if 'Quarterly_Net_New_ARR' in latest_row:
            net_new_arr = latest_row['Quarterly_Net_New_ARR']
        elif 'Net New ARR' in latest_row:
            net_new_arr = latest_row['Net New ARR']
        elif 'Net_New_ARR' in latest_row:
            net_new_arr = latest_row['Net_New_ARR']
        
        if current_arr is None or net_new_arr is None:
            return JSONResponse(status_code=400, content={
                "error": "CSV must contain ARR_End_of_Quarter (or cARR/ARR) and Quarterly_Net_New_ARR (or Net New ARR/Net_New_ARR) columns"
            })
        
        # Calculate growth rate
        growth_rate = (net_new_arr / current_arr) * 100 if current_arr > 0 else 0
        
        # Use the same logic as guided_forecast
        from financial_prediction import load_trained_model, predict_future_arr
        
        # Initialize guided system
        guided_system = EnhancedGuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Build primary inputs (exactly as in guided_forecast)
        primary_inputs = {
            'cARR': current_arr,
            'Net New ARR': net_new_arr,
            'ARR YoY Growth (in %)': growth_rate,
            'Quarter Num': 1
        }
        
        # Infer secondary metrics using the guided system
        inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
        
        # Add company name and quarter
        inferred_metrics['id_company'] = 'CSV Upload Company'
        inferred_metrics['Financial Quarter'] = 'FY24 Q1'
        
        # Create forecast-ready DataFrame
        forecast_df = guided_system.create_forecast_input(inferred_metrics)
        
        # Try to make prediction with trained model
        try:
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                # Add uncertainty quantification
                from simple_uncertainty_prediction import predict_with_simple_uncertainty
                forecast_results = predict_with_simple_uncertainty(forecast_df, uncertainty_factor=0.1)
                model_used = "LightGBM Model with Uncertainty (±10%)"
                forecast_success = True
            else:
                raise Exception("No trained model available")
        except Exception as e:
            # Use fallback calculation
            from enhanced_prediction import EnhancedFinancialPredictor
            predictor = EnhancedFinancialPredictor()
            forecast_results = predictor._generate_fallback_forecast(inferred_metrics)
            model_used = "Fallback Calculation"
            forecast_success = False
        
        # Generate insights
        insights = {
            'size_category': 'Early Stage' if current_arr < 1e6 else 'Growth Stage' if current_arr < 10e6 else 'Scale Stage' if current_arr < 100e6 else 'Enterprise',
            'growth_insight': f"Growth rate: {growth_rate:.1f}%",
            'efficiency_insight': f"Magic Number: {inferred_metrics.get('Magic_Number', 0):.2f}"
        }
        
        return {
            "company_name": "CSV Upload Company",
            "input_metrics": inferred_metrics,
            "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
            "insights": insights,
            "model_used": model_used,
            "forecast_success": forecast_success,
            "message": "CSV forecast completed successfully using the same logic as guided forecast!"
        }
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"CSV forecast failed: {str(e)}"})

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Enhanced conversational endpoint for financial forecasting and analysis."""
    message = request.message.lower()
    name = request.name
    preferred_model = request.preferred_model
    history = request.history or []

    # Check for financial data in the message
    arr_pattern = r'(\d+(?:\.\d+)?)\s*(?:million|m|k|k)?\s*(?:arr|annual recurring revenue|revenue)'
    net_new_pattern = r'(\d+(?:\.\d+)?)\s*(?:million|m|k|k)?\s*(?:net new|new arr|quarterly)'
    
    # Extract ARR and Net New ARR from natural language
    arr_match = re.search(arr_pattern, request.message, re.IGNORECASE)
    net_new_match = re.search(net_new_pattern, request.message, re.IGNORECASE)
    
    # Check for specific forecasting requests
    forecast_keywords = ['forecast', 'predict', 'projection', 'growth', 'future', 'next quarter', '2025']
    is_forecast_request = any(keyword in message for keyword in forecast_keywords)
    
    # Check for feature questions
    feature_keywords = ['magic number', 'burn multiple', 'rule of 40', 'gross margin', 'headcount', 'customers', 'churn', 'expansion']
    is_feature_question = any(keyword in message for keyword in feature_keywords)
    
    # Check for CSV mentions
    is_csv_request = 'csv' in message or 'upload' in message or 'file' in message
    
    # If user provides financial data and wants a forecast
    if (arr_match or net_new_match) and is_forecast_request:
        try:
            # Extract financial data
            current_arr = None
            net_new_arr = None
            
            if arr_match:
                arr_value = float(arr_match.group(1))
                # Handle units (million, k, etc.)
                if 'million' in request.message.lower() or 'm' in request.message.lower():
                    current_arr = arr_value * 1000000
                elif 'k' in request.message.lower():
                    current_arr = arr_value * 1000
                else:
                    current_arr = arr_value
            
            if net_new_match:
                net_new_value = float(net_new_match.group(1))
                # Handle units
                if 'million' in request.message.lower() or 'm' in request.message.lower():
                    net_new_arr = net_new_value * 1000000
                elif 'k' in request.message.lower():
                    net_new_arr = net_new_value * 1000
                else:
                    net_new_arr = net_new_value
            
            # If we have both values, run the forecast
            if current_arr and net_new_arr:
                # Use the same logic as guided_forecast
                from financial_prediction import load_trained_model, predict_future_arr
                
                # Initialize guided system
                guided_system = EnhancedGuidedInputSystem()
                guided_system.initialize_from_training_data()
                
                # Build primary inputs
                growth_rate = (net_new_arr / current_arr) * 100 if current_arr > 0 else 0
                primary_inputs = {
                    'cARR': current_arr,
                    'Net New ARR': net_new_arr,
                    'ARR YoY Growth (in %)': growth_rate,
                    'Quarter Num': 1
                }
                
                # Infer secondary metrics
                inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
                inferred_metrics['id_company'] = 'Chat User'
                inferred_metrics['Financial Quarter'] = 'FY24 Q1'
                
                # Create forecast-ready DataFrame
                forecast_df = guided_system.create_forecast_input(inferred_metrics)
                            
                # Make prediction with uncertainty
                try:
                    trained_model = load_trained_model('lightgbm_financial_model.pkl')
                    if trained_model:
                        from simple_uncertainty_prediction import predict_with_simple_uncertainty
                        forecast_results = predict_with_simple_uncertainty(forecast_df, uncertainty_factor=0.1)
                        model_used = "LightGBM Model with Uncertainty (±10%)"
                    else:
                        raise Exception("Model not available")
                except Exception as e:
                    # Fallback calculation
                    from enhanced_prediction import EnhancedFinancialPredictor
                    predictor = EnhancedFinancialPredictor()
                    forecast_results = predictor._generate_fallback_forecast(inferred_metrics)
                    model_used = "Fallback Calculation"
                
                # Generate conversational analysis
                analysis_prompt = f"""
                Based on the financial data provided:
                - Current ARR: ${current_arr:,.0f}
                - Net New ARR: ${net_new_arr:,.0f}
                - Growth Rate: {growth_rate:.1f}%
                
                And the forecast results showing growth predictions for the next 4 quarters, provide a conversational, helpful analysis that includes:
                1. What the numbers mean in simple terms
                2. Key insights about the company's stage and performance
                3. Potential risks or opportunities
                4. Suggestions for improvement
                5. How the uncertainty ranges affect the forecast
                
                Make it conversational and easy to understand for a business person.
                """
                
                analysis = llm.invoke(analysis_prompt).content
                
                return {
                    "response": analysis,
                    "data": {
                        "model": model_used,
                        "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
                        "input_metrics": inferred_metrics,
                        "message": "Forecast completed successfully via chat!"
                    }
                }
            else:
                # Ask for missing data
                missing_data = []
                if not current_arr:
                    missing_data.append("current ARR")
                if not net_new_arr:
                    missing_data.append("net new ARR")
                
                return {
                    "response": f"I'd be happy to help with your forecast! I need a bit more information. Could you please provide your {', '.join(missing_data)}? You can say something like 'My ARR is $2.1M and net new ARR is $320K'."
                }
                
        except Exception as e:
            return {"response": f"I encountered an error while processing your request: {str(e)}. Could you try rephrasing your financial data?"}
    
    # If user asks about specific features
    elif is_feature_question:
        feature_explanations = {
            'magic number': "The Magic Number measures sales efficiency: Net New ARR ÷ Sales & Marketing spend. Above 1.0 is excellent, 0.5-1.0 is good, below 0.5 needs improvement.",
            'burn multiple': "Burn Multiple shows cash efficiency: Net Burn ÷ Net New ARR. Below 1.0 is great, 1.0-2.0 is acceptable, above 2.0 is concerning.",
            'rule of 40': "Rule of 40 combines growth + profitability: Growth Rate + Profit Margin. Should be ≥40% for healthy SaaS companies.",
            'gross margin': "Gross Margin = (Revenue - COGS) ÷ Revenue. SaaS companies typically aim for 70-90% gross margins.",
            'headcount': "Employee count affects operational efficiency and burn rate. ARR per headcount is a key efficiency metric.",
            'customers': "Customer count and expansion/churn rates are crucial for growth sustainability.",
            'churn': "Customer churn rate should typically be <5% annually for healthy SaaS companies.",
            'expansion': "Expansion revenue from existing customers is often more efficient than new customer acquisition."
        }
        
        # Find which feature they're asking about
        for keyword, explanation in feature_explanations.items():
            if keyword in message:
                return {
                    "response": f"Great question about {keyword}! {explanation} Would you like me to help you calculate this metric for your company?"
                }
        
        return {
            "response": "I can help explain any SaaS metrics! Which specific metric would you like to know more about? Common ones include Magic Number, Burn Multiple, Rule of 40, Gross Margin, and more."
        }
    
    # If user mentions CSV
    elif is_csv_request:
        return {
            "response": "I can help you analyze CSV data! You have a few options:\n\n1. **Upload CSV**: Use the `/predict_csv` endpoint to upload your file\n2. **Chat with data**: Tell me your ARR and net new ARR directly in chat\n3. **Ask questions**: I can explain any SaaS metrics or help with analysis\n\nWhat would you prefer?"
        }
    
    # General conversation
    else:
        greeting = f"Hi {name}!" if name else "Hi!"
        project_info = GPT_INFO.get("project_info", {})
        project_info_str = " ".join([
            f"Creator: {project_info.get('creator', '')}.",
            f"Project name: {project_info.get('project_name', '')}.",
            f"Purpose: {project_info.get('purpose', '')}.",
            f"Contact: {project_info.get('contact', '')}."
        ])
        
        # Get macro indicators
        gprh = gprh_trend_analysis()
        vix = vix_trend_analysis()
        move = move_trend_analysis()
        
        system_prompt = f"""
        {greeting} I'm your AI financial forecasting assistant! I can help you with:
        
        📊 **Financial Forecasting**: Just tell me your ARR and net new ARR, and I'll predict your growth
        📈 **SaaS Metrics Analysis**: Ask about Magic Number, Burn Multiple, Rule of 40, etc.
        📋 **CSV Analysis**: Upload your financial data for detailed analysis
        🌍 **Market Insights**: Get current macro indicators (GPRH: {gprh['traffic_light']}, VIX: {vix['traffic_light']}, MOVE: {move['traffic_light']})
        
        Project info: {project_info_str}
        
        How can I help you today? You can:
        - Say "My ARR is $2.1M and net new ARR is $320K" for a forecast
        - Ask "What is Magic Number?" for metric explanations
        - Ask about market conditions or macro trends
        - Or just chat naturally!
        """
        
        conversation = [{"role": "system", "content": system_prompt}]
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
def guided_forecast(request: EnhancedGuidedInputRequest):
    """Guided forecasting with intelligent defaults - using the original working logic."""
    try:
        from financial_prediction import load_trained_model, predict_future_arr
        
        # Initialize guided system
        guided_system = EnhancedGuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Build primary inputs (exactly as in the working version)
        growth_rate = (request.net_new_arr / request.current_arr * 100) if request.current_arr > 0 else 0
        primary_inputs = {
            'cARR': request.current_arr,
            'Net New ARR': request.net_new_arr,
            'ARR YoY Growth (in %)': growth_rate,
            'Quarter Num': 1
        }
        
        # Infer secondary metrics using the guided system (exactly as in working version)
        inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
        
        # Apply enhanced mode overrides if enabled
        if request.enhanced_mode:
            print(f"🔧 Enhanced mode enabled - applying user-provided categorical data")
            
            # Map enhanced inputs to internal field names
            enhanced_mapping = {
                'sector': 'Sector',
                'country': 'Country', 
                'currency': 'Currency'
            }
            
            # Apply enhanced overrides for non-None values
            for enhanced_key, value in request.dict().items():
                if value is not None and enhanced_key in enhanced_mapping:
                    field_name = enhanced_mapping[enhanced_key]
                    inferred_metrics[field_name] = value
                    print(f"🔧 Enhanced override: {field_name} = {value}")
        
        # Apply advanced mode overrides if provided
        advanced_metrics_dict = {}
        if request.advanced_mode and request.advanced_metrics:
            # Map advanced metrics to the expected field names
            advanced_mapping = {
                'sales_marketing': 'Sales & Marketing',
                'ebitda': 'EBITDA',
                'cash_burn': 'Cash Burn (OCF & ICF)',
                'rule_of_40': 'LTM Rule of 40% (ARR)',
                'arr_yoy_growth': 'ARR YoY Growth (in %)',
                'revenue_yoy_growth': 'Revenue YoY Growth (in %)',
                'magic_number': 'Magic_Number',
                'burn_multiple': 'Burn_Multiple',
                'customers_eop': 'Customers (EoP)',
                'expansion_upsell': 'Expansion & Upsell',
                'churn_reduction': 'Churn & Reduction',
                'gross_margin': 'Gross Margin (in %)',
                'headcount': 'Headcount (HC)',
                'net_profit_margin': 'Net Profit/Loss Margin (in %)'
            }
            
            # Apply overrides for non-None values
            for adv_key, value in request.advanced_metrics.dict().items():
                if value is not None and adv_key in advanced_mapping:
                    field_name = advanced_mapping[adv_key]
                    inferred_metrics[field_name] = value
                    advanced_metrics_dict[field_name] = value
                    print(f"🔧 Advanced override: {field_name} = {value}")
        
        # Add company name and quarter (exactly as in working version)
        inferred_metrics['id_company'] = request.company_name or 'Anonymous Company'
        inferred_metrics['Financial Quarter'] = 'FY24 Q1'
        
        # Create forecast-ready DataFrame using the enhanced guided system
        if request.historical_arr and all([request.historical_arr.q1_arr, request.historical_arr.q2_arr, 
                                         request.historical_arr.q3_arr, request.historical_arr.q4_arr]):
            # Use historical ARR data if provided
            historical_arr = [
                request.historical_arr.q1_arr,
                request.historical_arr.q2_arr,
                request.historical_arr.q3_arr,
                request.historical_arr.q4_arr
            ]
            forecast_df = guided_system.create_forecast_input_with_history(
                current_arr=request.current_arr,
                net_new_arr=request.net_new_arr,
                historical_arr=historical_arr,
                advanced_metrics=advanced_metrics_dict if request.advanced_mode else None
            )
        else:
            # Use minimal inputs
            forecast_df = guided_system.create_forecast_input_with_history(
                current_arr=request.current_arr,
                net_new_arr=request.net_new_arr,
                advanced_metrics=advanced_metrics_dict if request.advanced_mode else None
            )
        
        # Try to make prediction with trained model (exactly as in working version)
        try:
            trained_model = load_trained_model('lightgbm_financial_model.pkl')
            if trained_model:
                # Add uncertainty quantification
                from simple_uncertainty_prediction import predict_with_simple_uncertainty
                forecast_results = predict_with_simple_uncertainty(forecast_df, uncertainty_factor=0.1)
                model_used = "LightGBM Model with Uncertainty (±10%)"
                forecast_success = True
            else:
                raise Exception("No trained model available")
        except Exception as e:
            # Use fallback calculation (exactly as in working version)
            from enhanced_prediction import EnhancedFinancialPredictor
            predictor = EnhancedFinancialPredictor()
            forecast_results = predictor._generate_fallback_forecast(inferred_metrics)
            model_used = "Fallback Calculation"
            forecast_success = False
        
        # Generate insights (exactly as in working version)
        insights = {
            'size_category': 'Early Stage' if request.current_arr < 1e6 else 'Growth Stage' if request.current_arr < 10e6 else 'Scale Stage' if request.current_arr < 100e6 else 'Enterprise',
            'growth_insight': f"Growth rate: {primary_inputs['ARR YoY Growth (in %)']:.1f}%",
            'efficiency_insight': f"Magic Number: {inferred_metrics.get('Magic_Number', 0):.2f}"
        }
        
        return {
            "company_name": request.company_name or "Anonymous Company",
            "input_metrics": inferred_metrics,
            "forecast_results": forecast_results.to_dict('records') if hasattr(forecast_results, 'to_dict') else forecast_results,
            "insights": insights,
            "model_used": model_used,
            "forecast_success": forecast_success,
            "message": "Guided forecast completed successfully using the original working logic!"
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
            "guided_forecast": "POST /guided_forecast - Guided input with intelligent defaults + Advanced Mode",
            "makro_analysis": "GET /makro_analysis - Macroeconomic indicators"
        },
        "features": {
            "guided_input": "Only need ARR + Net New ARR, intelligently infers the rest",
            "enhanced_mode": "Optional sector/country/currency selection for better accuracy",
            "historical_arr": "Provide 4 quarters of historical ARR data for better predictions",
            "advanced_mode": "Override any of 14 key metrics with your own values",
            "uncertainty_quantification": "±10% uncertainty bands on all predictions",
            "adaptive_defaults": "Uses training data relationships for realistic estimates"
        },
        "advanced_metrics": [
            "sales_marketing", "ebitda", "cash_burn", "rule_of_40", 
            "arr_yoy_growth", "revenue_yoy_growth", "magic_number", 
            "burn_multiple", "customers_eop", "expansion_upsell", 
            "churn_reduction", "gross_margin", "headcount", "net_profit_margin"
        ],
        "enhanced_mode_options": {
            "sectors": VALID_SECTORS,
            "countries": VALID_COUNTRIES,
            "currencies": VALID_CURRENCIES
        }
    } 