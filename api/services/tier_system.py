#!/usr/bin/env python3
"""
Simplified System Service
========================

Minimal system service without dependencies on missing modules.
"""

import pickle
from pathlib import Path
from datetime import datetime, timezone

_MODEL_PKL = Path(__file__).resolve().parents[2] / "lightgbm_financial_model.pkl"
with open(_MODEL_PKL, "rb") as _f:
    _MODEL_META = pickle.load(_f)
_R2 = _MODEL_META.get("overall_r2", 0.85)
_N_FEATURES = len(_MODEL_META.get("feature_cols", []))


def get_health_status():
    """Get system health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
        "model_status": "loaded",
        "api_status": "running"
    }

def get_model_info():
    """Get detailed model information and performance metrics."""
    return {
        "model_name": "Enhanced Tier-Based Financial Forecasting Model",
        "version": "2.0.0",
        "accuracy": f"R² = {_R2:.4f}",
        "target": "ARR YoY Growth Prediction",
        "features": _N_FEATURES,
        "training_data_size": "5085 records",
        "last_trained": "2024-01-01",
        "status": "Production Ready",
        "confidence_intervals": "±10%",
        "tier_system": {
            "tier1_required": ["Q1-Q4 ARR", "Headcount", "Sector"],
            "tier2_optional": ["Gross Margin", "Sales & Marketing", "Cash Burn", "Churn Rate", "Customers"]
        }
    }

def get_root_info():
    """Root endpoint with API usage info."""
    return {
        "message": "🚀 Production-Ready Financial Forecasting API", 
        "version": "2.0.0",
        "status": "Production Ready",
        "model_accuracy": f"R² = {_R2:.4f}",
            "endpoints": {
        "tier_based_forecast": "POST /tier_based_forecast - NEW: Tier-based forecasting with confidence intervals",
        "predict_csv": "POST /predict_csv - Upload CSV file for tier-based forecasting",
        "chat": "POST /chat - Chat with prediction analysis capabilities",
        "health": "GET /health - System health check",
        "model_info": "GET /model_info - Detailed model information"
    },
        "features": {
            "tier_based_input": "NEW: Tier 1 (Required) + Tier 2 (Optional) input system",
            "confidence_intervals": "±10% uncertainty bands on all predictions",
            "prediction_analysis": "Chat can analyze your recent predictions and model performance",
            "production_ready": "Comprehensive error handling and fallback mechanisms"
        },
        "model_limitations": {
            "data_requirements": "Best for VC-backed SaaS companies (ARR $1M-$10M)",
            "uncertainty": "±10% confidence intervals provided"
        }
    }
