import numpy as np
import pandas as pd
import joblib
from typing import List
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import Tool as LC_Tool
from prediction_analysis_tools import prediction_analysis_tool
import re
import json
from datetime import datetime
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# from enhanced_guided_input import EnhancedGuidedInputSystem  # File doesn't exist
# from production_ready_system import get_production_forecaster  # File doesn't exist
from api.models.schemas import FeatureInput, ChatRequest, EnhancedGuidedInputRequest, TierBasedRequest

load_dotenv()

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

# --- Utility Functions ---
def parse_features(input_str: str):
    """Parse a comma-separated string of features into a list of floats."""
    try:
        return [float(x.strip()) for x in input_str.split(',')]
    except Exception:
        return None


# Old CSV tool removed - now using tier-based CSV processing

# --- LangChain Tools ---
arr_growth_tool_lgbm = LC_Tool(
    name="LightGBM ARR Growth Predictor",
    func=lambda s: arr_tool(s, 'lightgbm'),
    description="Predicts ARR YoY growth for the next 4 quarters using the LightGBM model. Input: 28 comma-separated features."
)
# Old CSV LangChain tool removed - now using tier-based CSV processing

# --- LangChain Agent ---
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
agent = initialize_agent(
    tools=[arr_growth_tool_lgbm, prediction_analysis_tool],  # Added prediction analysis tool
    llm=llm,
    agent_type="chat-zero-shot-react-description",
    verbose=True
)

# Old predict_from_csv function removed - now using tier_prediction.py

def handle_chat(request: ChatRequest):
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
    
    # Extract enhanced mode parameters (sector, country, currency)
    sector_pattern = r'(?:sector|industry):\s*([a-zA-Z\s&]+)'
    country_pattern = r'(?:country|location):\s*([a-zA-Z\s]+)'
    currency_pattern = r'(?:currency):\s*([A-Z]{3})'
    
    sector_match = re.search(sector_pattern, request.message, re.IGNORECASE)
    country_match = re.search(country_pattern, request.message, re.IGNORECASE)
    currency_match = re.search(currency_pattern, request.message, re.IGNORECASE)
    
    # Check for specific forecasting requests
    forecast_keywords = ['forecast', 'predict', 'projection', 'growth', 'future', 'next quarter', '2025']
    is_forecast_request = any(keyword in message for keyword in forecast_keywords)
    
    # Check for feature questions
    feature_keywords = ['magic number', 'burn multiple', 'rule of 40', 'gross margin', 'headcount', 'customers', 'churn', 'expansion']
    is_feature_question = any(keyword in message for keyword in feature_keywords)
    
    # Check for CSV mentions (redirected to tier-based system)
    is_csv_request = 'csv' in message or 'upload' in message or 'file' in message
    
    # Check for algorithm/model explanation requests (check this first to avoid conflicts)
    algorithm_keywords = ['how does', 'how it works', 'algorithm', 'model works', 'feature injection', 'intelligent feature', 'tier based', 'lightgbm', 'gradient boosted']
    is_algorithm_question = any(keyword in message.lower() for keyword in algorithm_keywords)

    
    # Check for prediction analysis requests (exclude algorithm-related terms)
    prediction_keywords = ['prediction', 'forecast', 'analysis', 'accuracy', 'confidence', 'growth pattern', 'performance']
    is_prediction_analysis = any(keyword in message.lower() for keyword in prediction_keywords) and not is_algorithm_question
    
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
                
                # Check for enhanced mode (sector, country, currency)
                enhanced_mode = sector_match or country_match or currency_match
                enhanced_metrics = {}
                
                if enhanced_mode:
                    if sector_match:
                        enhanced_metrics['sector'] = sector_match.group(1).strip()
                    if country_match:
                        enhanced_metrics['country'] = country_match.group(1).strip()
                    if currency_match:
                        enhanced_metrics['currency'] = currency_match.group(1).strip()
                
                # Build primary inputs
                growth_rate = (net_new_arr / current_arr) * 100 if current_arr > 0 else 0
                primary_inputs = {
                    'cARR': current_arr,
                    'Net New ARR': net_new_arr,
                    'ARR YoY Growth (in %)': growth_rate,
                    'Quarter Num': 1
                }
                
                # Infer secondary metrics
                inferred_metrics = guided_system._infer_secondary_metrics(current_arr, net_new_arr, growth_rate)
                inferred_metrics['id_company'] = 'Chat User'
                inferred_metrics['Financial Quarter'] = 'FY24 Q1'
                
                # Create forecast-ready DataFrame with historical data (same as guided_forecast)
                forecast_df = guided_system.create_forecast_input_with_history(
                    current_arr=current_arr,
                    net_new_arr=net_new_arr,
                    enhanced_mode=enhanced_mode,
                    enhanced_metrics=enhanced_metrics if enhanced_mode else None
                )
                            
                # Make prediction with uncertainty
                try:
                    # Use the correct single-quarter models
                    import pickle
                    with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
                        model_data = pickle.load(f)
                    
                    if model_data:
                        # Use the final solution approach
                        from test_final_solution import predict_with_documented_bias
                        forecast_results = predict_with_documented_bias(model_data, forecast_df, "Chat User")
                        model_used = "High-Accuracy Single-Quarter Models (R² = 0.7966)"
                    else:
                        raise Exception("Model not available")
                except Exception as e:
                    # Fallback calculation
                    # from enhanced_guided_input import EnhancedGuidedInputSystem  # File doesn't exist
                    enhanced_system = EnhancedGuidedInputSystem()
                    enhanced_system.initialize_from_training_data()
                    forecast_results = enhanced_system.create_forecast_input_with_history(
                        current_arr=inferred_metrics['cARR'],
                        net_new_arr=inferred_metrics['Net New ARR']
                    )
                    model_used = "Fallback Calculation"
                
                # Generate conversational analysis
                enhanced_info = ""
                if enhanced_mode:
                    enhanced_info = f"\nEnhanced context: {', '.join([f'{k}: {v}' for k, v in enhanced_metrics.items()])}"
                
                analysis_prompt = f"""
                Based on the financial data provided:
                - Current ARR: ${current_arr:,.0f}
                - Net New ARR: ${net_new_arr:,.0f}
                - Growth Rate: {growth_rate:.1f}%{enhanced_info}
                
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
    
    # If user asks about how the algorithm/model works (check this early)
    elif is_algorithm_question:

        algorithm_info = GPT_INFO.get("algorithm_explanation", {})
        model_info = GPT_INFO.get("model_details", {})
        
        response = f"🤖 **How Our Enhanced Tier-Based Prediction System Works**\n\n"
        response += f"**Overview:** {algorithm_info.get('overview', 'Our system uses advanced machine learning to predict ARR growth.')}\n\n"
        
        response += f"**🎯 Stage 1: Tier-Based Input**\n"
        stage1 = algorithm_info.get('stage_1_tier_based_input', {})
        response += f"• {stage1.get('description', 'Minimal required data with intelligent defaults')}\n"
        response += f"• {stage1.get('benefit', 'Reduces user burden while maintaining accuracy')}\n\n"
        
        response += f"**🧠 Stage 2: Intelligent Feature Completion**\n"
        stage2 = algorithm_info.get('stage_2_intelligent_feature_completion', {})
        response += f"• {stage2.get('description', 'Advanced pattern matching to infer missing features')}\n"
        response += f"• **Process:** {' → '.join(stage2.get('process', ['Company profiling', 'Similarity matching', 'Feature inference']))}\n"
        response += f"• **Features Created:** {len(stage2.get('features_created', []))}+ engineered features\n\n"
        
        response += f"**⚡ Stage 3: LightGBM Modeling**\n"
        stage3 = algorithm_info.get('stage_3_modeling', {})
        response += f"• **Algorithm:** {stage3.get('algorithm', 'LightGBM')}\n"
        response += f"• **Why LightGBM:** {stage3.get('why_lightgbm', 'Excellent for tabular data and complex relationships')}\n"
        response += f"• **Training Data:** {stage3.get('training_data', '500+ VC-backed companies')}\n"
        response += f"• **Features:** {stage3.get('feature_count', '152')} engineered features per prediction\n\n"
        
        response += f"**📊 Confidence Intervals**\n"
        confidence = algorithm_info.get('confidence_intervals', {})
        response += f"• **Method:** {confidence.get('method', '±10% uncertainty bands')}\n"
        response += f"• **Rationale:** {confidence.get('rationale', 'Based on model performance and business uncertainty')}\n\n"
        
        response += f"**🎯 Key Features:**\n"
        features = stage2.get('features_created', [])
        for feature in features[:4]:  # Show first 4 features
            response += f"• {feature}\n"
        
        response += f"\n**📈 Performance:** R² = {model_info.get('performance', {}).get('r2', 0.7966):.1%} with {model_info.get('performance', {}).get('confidence_intervals', '±10%')} confidence intervals"
        
        return {"response": response}
    
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
    

    
    # If user asks about predictions/analysis
    elif is_prediction_analysis:
        try:
            # Use the LangChain agent with prediction analysis tools
            response = agent.run(f"User is asking about predictions: {message}")
            return {"response": response}
        except Exception as e:
            return {
                "response": f"I can help analyze your predictions! I can provide:\n\n📊 **Prediction Analysis**: Summary of recent forecasts\n🏢 **Company Analysis**: Detailed breakdown for specific companies\n📈 **Growth Patterns**: Analysis of growth trends across predictions\n🎯 **Model Performance**: Accuracy and confidence metrics\n📋 **Confidence Intervals**: Uncertainty analysis\n\nTry asking: 'Show me my recent predictions' or 'Analyze the growth patterns' or 'What's the model accuracy?'"
            }
    
    # If user mentions CSV (redirect to tier-based system)
    elif is_csv_request:
        return {
            "response": "I can help you analyze CSV data! Please use the new tier-based CSV endpoint:\n\n**Upload CSV**: Use the `/predict_csv` endpoint in the tier-based system for enhanced analysis with confidence intervals and intelligent feature completion.\n\nThis new system provides:\n- Intelligent sector inference\n- Advanced feature completion\n- Confidence intervals (±10%)\n- Better accuracy (R² = 79.66%)\n\nOr you can tell me your ARR and net new ARR directly in chat for a quick analysis!"
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
        
        from api.services.macro_analysis import get_macro_analysis
        macro_data = get_macro_analysis()
        gprh = macro_data.get('gprh', {'traffic_light': 'Unknown'})
        vix = macro_data.get('vix', {'traffic_light': 'Unknown'})
        move = macro_data.get('move', {'traffic_light': 'Unknown'})
        bvp = macro_data.get('bvp', {'traffic_light': 'Unknown'})
        
        system_prompt = f"""
        {greeting} I'm your AI financial forecasting assistant! I can help you with:
        
        📊 **Financial Forecasting**: Just tell me your ARR and net new ARR, and I'll predict your growth
        🌟 **Enhanced Mode**: Add sector, country, currency for better predictions (e.g., "ARR $2.1M, net new ARR $320K, sector: SaaS, country: United States")
        📈 **SaaS Metrics Analysis**: Ask about Magic Number, Burn Multiple, Rule of 40, etc.
        📋 **CSV Analysis**: Use the new tier-based `/predict_csv` endpoint for enhanced analysis
        🔍 **Prediction Analysis**: Ask me about your recent predictions, model performance, growth patterns, and confidence intervals
        🤖 **Algorithm Explanation**: Ask "How does the model work?" to learn about our 3-stage tier-based system with intelligent feature completion
        🌍 **Market Insights**: Get current macro indicators (GPRH: {gprh['traffic_light']}, VIX: {vix['traffic_light']}, MOVE: {move['traffic_light']}, BVP: {bvp['traffic_light']})
        
        Project info: {project_info_str}
        
        How can I help you today? You can:
        - Say "My ARR is $2.1M and net new ARR is $320K" for a forecast
        - Add context: "ARR $2.1M, net new ARR $320K, sector: Data & Analytics, country: United States"
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

def perform_tier_based_forecast(request: TierBasedRequest):
    """Tier-based forecasting using the new enhanced model with confidence intervals."""
    try:
        # Import the tier-based system
        from tier_based_prediction_system import TierBasedPredictionSystem
        
        # Prepare Tier 1 data
        tier1_data = {
            'q1_arr': request.q1_arr,
            'q2_arr': request.q2_arr,
            'q3_arr': request.q3_arr,
            'q4_arr': request.q4_arr,
            'headcount': request.headcount,
            'sector': request.sector
        }
        
        # Prepare Tier 2 data if provided
        tier2_data = None
        if request.tier2_metrics:
            tier2_data = {
                'gross_margin': request.tier2_metrics.gross_margin,
                'sales_marketing': request.tier2_metrics.sales_marketing,
                'cash_burn': request.tier2_metrics.cash_burn,
                'customers': request.tier2_metrics.customers,
                'churn_rate': request.tier2_metrics.churn_rate,
                'expansion_rate': request.tier2_metrics.expansion_rate
            }
        
        # Initialize tier-based system
        tier_system = TierBasedPredictionSystem()
        
        # Get predictions
        predictions, company_df = tier_system.predict_with_tiers(tier1_data, tier2_data)
        
        # Format response
        forecast_results = []
        for pred in predictions:
            forecast_results.append({
                "quarter": pred['Quarter'],
                "predicted_arr": pred['ARR'],
                "pessimistic_arr": pred['Pessimistic_ARR'],
                "optimistic_arr": pred['Optimistic_ARR'],
                "yoy_growth_percent": pred['YoY_Growth_Percent'],
                "confidence_interval": f"±10%"
            })
        
        # Calculate insights
        current_arr = request.q4_arr
        final_predicted_arr = predictions[-1]['ARR']
        total_growth = ((final_predicted_arr - current_arr) / current_arr) * 100
        
        insights = {
            'company_name': request.company_name or "Anonymous Company",
            'current_arr': current_arr,
            'predicted_final_arr': final_predicted_arr,
            'total_growth_percent': total_growth,
            'tier_used': 'Tier 1 + Tier 2' if tier2_data else 'Tier 1 Only',
            'model_accuracy': 'R² = 0.7966 (79.66%)',
            'confidence_intervals': '±10% on all predictions'
        }
        
        return {
            "success": True,
            "company_name": request.company_name or "Anonymous Company",
            "insights": insights,
            "forecast": forecast_results,
            "model_used": "Enhanced Tier-Based Model with Confidence Intervals",
            "tier_analysis": {
                "tier1_provided": True,
                "tier2_provided": tier2_data is not None,
                "tier2_metrics_count": len([v for v in (tier2_data or {}).values() if v is not None]) if tier2_data else 0
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Tier-based forecast failed: {str(e)}",
            "company_name": request.company_name or "Anonymous Company"
        }

def perform_guided_forecast(request: EnhancedGuidedInputRequest):
    """Guided forecasting with intelligent defaults - using the original working logic."""
    if request.current_arr <= 0:
        raise ValueError("Current ARR must be greater than 0")
    
    if request.net_new_arr < 0:
        raise ValueError("Net New ARR cannot be negative")
    
    if request.current_arr < request.net_new_arr:
        raise ValueError("Net New ARR cannot be greater than Current ARR")
    
    growth_rate = (request.net_new_arr / request.current_arr * 100) if request.current_arr > 0 else 0
    if growth_rate > 1000:
        raise ValueError("Growth rate too high (>1000%). Please check your inputs.")

    guided_system = EnhancedGuidedInputSystem()
    guided_system.initialize_from_training_data()
    
    inferred_metrics = guided_system._infer_secondary_metrics(request.current_arr, request.net_new_arr, growth_rate)
    
    # ... (rest of the guided forecast logic) ...
    
    production_forecaster = get_production_forecaster()
    prediction_result = production_forecaster.predict_with_fallback(
        current_arr=request.current_arr,
        net_new_arr=request.net_new_arr,
        company_name=request.company_name or "Anonymous Company",
        historical_arr=request.historical_arr.dict() if request.historical_arr else None
    )
    
    if not prediction_result["success"]:
        raise Exception(prediction_result["error"])
        
    insights = {
        'size_category': 'Early Stage' if request.current_arr < 1e6 else 'Growth Stage' if request.current_arr < 10e6 else 'Scale Stage' if request.current_arr < 100e6 else 'Enterprise',
        'growth_insight': f"Growth rate: {growth_rate:.1f}%",
        'efficiency_insight': f"Magic Number: {inferred_metrics.get('Magic_Number', 0):.2f}"
    }
    
    return {
        "company_name": request.company_name or "Anonymous Company",
        "input_metrics": inferred_metrics,
        "forecast_results": prediction_result["forecast_results"].to_dict('records') if hasattr(prediction_result["forecast_results"], 'to_dict') else prediction_result["forecast_results"],
        "insights": insights,
        "model_used": prediction_result["model_used"],
        "forecast_success": True,
        "message": "Guided forecast completed successfully using the original working logic!"
    }
