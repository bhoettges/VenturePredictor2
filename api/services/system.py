import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from production_ready_system import get_production_forecaster
from api.models.schemas import VALID_SECTORS, VALID_COUNTRIES, VALID_CURRENCIES

def get_health_status():
    """System health check."""
    try:
        production_forecaster = get_production_forecaster()
        status = production_forecaster.get_system_status()
        
        status["api_status"] = "Healthy"
        status["timestamp"] = datetime.now().isoformat()
        
        model_exists = Path("lightgbm_single_quarter_models.pkl").exists()
        status["model_file_exists"] = model_exists
        
        if status["model_loaded"] and model_exists:
            status["overall_health"] = "Healthy"
        elif model_exists:
            status["overall_health"] = "Degraded (Model not loaded)"
        else:
            status["overall_health"] = "Critical (Model file missing)"
        
        return status
        
    except Exception as e:
        return {
            "api_status": "Error",
            "overall_health": "Critical",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_model_info():
    """Get detailed model information and performance metrics."""
    production_forecaster = get_production_forecaster()
    status = production_forecaster.get_system_status()
    
    model_info = {
        "model_type": "LightGBM Single-Quarter Models",
        "target": "ARR YoY Growth Prediction",
        "horizon": "4 quarters",
        "performance": status.get("model_performance", {}),
        "overall_r2": status.get("overall_r2", 0),
        "model_limitations": {
            "q1_bias": "Q1 predictions tend to be 7-8% optimistic",
            "uncertainty": "±10% confidence intervals provided",
            "data_requirements": "Model trained on VC-backed companies (ARR 1M-10M range)",
            "bias_correction": "Q1 bias is documented but not corrected to maintain high accuracy"
        },
        "recommended_usage": {
            "best_for": "VC-backed SaaS companies with ARR between $1M-$10M",
            "input_requirements": "Current ARR and Net New ARR (minimum)",
            "output_format": "4-quarter forecast with confidence intervals",
            "accuracy": "R² = 0.7966 (79.66% accuracy)"
        }
    }
    
    return model_info

def get_root_info():
    """Root endpoint with API usage info."""
    return {
        "message": "🚀 Production-Ready Financial Forecasting API", 
        "version": "2.0.0",
        "status": "Production Ready",
        "model_accuracy": "R² = 0.7966 (79.66%)",
        "endpoints": {
            "predict_raw": "POST /predict_raw - Predict from raw features",
            "predict_csv": "POST /predict_csv - Predict from CSV upload",
            "chat": "POST /chat - Conversational AI interface",
            "guided_forecast": "POST /guided_forecast - Guided input with intelligent defaults + Advanced Mode",
            "tier_based_forecast": "POST /tier_based_forecast - NEW: Tier-based forecasting with confidence intervals",
            "makro_analysis": "GET /makro_analysis - Macroeconomic indicators",
            "health": "GET /health - System health check",
            "model_info": "GET /model_info - Detailed model information"
        },
        "features": {
            "guided_input": "Only need ARR + Net New ARR, intelligently infers the rest",
            "enhanced_mode": "Optional sector/country/currency selection for better accuracy",
            "historical_arr": "Provide 4 quarters of historical ARR data for better predictions",
            "advanced_mode": "Override any of 14 key metrics with your own values",
            "tier_based_input": "NEW: Tier 1 (Required) + Tier 2 (Optional) input system",
            "confidence_intervals": "±10% uncertainty bands on all predictions",
            "adaptive_defaults": "Uses training data relationships for realistic estimates",
            "production_ready": "Comprehensive error handling and fallback mechanisms",
            "documented_limitations": "Transparent about Q1 bias and model limitations"
        },
        "model_limitations": {
            "q1_bias": "Q1 predictions tend to be 7-8% optimistic (documented)",
            "data_requirements": "Best for VC-backed SaaS companies (ARR $1M-$10M)",
            "uncertainty": "±10% confidence intervals provided"
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
        },
        "documentation": {
            "github": "https://github.com/your-repo/financial-forecasting",
            "contact": "balthasar@hoettges.io",
            "support": "support@ventureprophet.com"
        }
    }
