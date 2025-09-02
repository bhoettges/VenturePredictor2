from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import sys
import pandas as pd
import io
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.services.tier_prediction import perform_tier_based_forecast, predict_from_csv
from api.services.prediction import handle_chat
from api.models.schemas import TierBasedRequest, ChatRequest

router = APIRouter()

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

@router.post("/predict_csv")
def predict_csv(file: UploadFile = File(...), company_name: str = Form(None)):
    """Predict from CSV upload using the tier-based system."""
    try:
        # Read the uploaded file
        file_content = file.file.read()
        
        # Process the CSV
        result = predict_from_csv(file_content, company_name)
        
        if result["success"]:
            return result
        else:
            return JSONResponse(status_code=400, content={"error": result["error"]})
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"CSV upload failed: {str(e)}"})

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Chat endpoint with prediction analysis capabilities."""
    try:
        response = handle_chat(request)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": f"I encountered an error: {str(e)}"})
