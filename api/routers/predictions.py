from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.services.prediction import (
    handle_chat, 
    perform_guided_forecast,
    perform_tier_based_forecast
)
from api.models.schemas import FeatureInput, ChatRequest, EnhancedGuidedInputRequest, TierBasedRequest

router = APIRouter()

# @router.post("/predict_raw")
# def predict_raw(data: FeatureInput):
#     try:
#         result = predict_raw_features(data)
#         return result
#     except ValueError as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

# Old CSV endpoint removed - now using /predict_csv in tier_predictions.py

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        response = handle_chat(request)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": f"I encountered an error: {str(e)}"})

@router.post("/guided_forecast")
def guided_forecast(request: EnhancedGuidedInputRequest):
    try:
        result = perform_guided_forecast(request)
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Guided forecast failed: {str(e)}"})

@router.post("/tier_based_forecast")
def tier_based_forecast(request: TierBasedRequest):
    """New tier-based forecasting endpoint with confidence intervals."""
    try:
        result = perform_tier_based_forecast(request)
        if result["success"]:
            return result
        else:
            return JSONResponse(status_code=400, content={"error": result["error"]})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Tier-based forecast failed: {str(e)}"})
