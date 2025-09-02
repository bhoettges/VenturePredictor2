#!/usr/bin/env python3
"""
Production-Ready Financial Forecasting System
Comprehensive error handling, validation, and monitoring
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import traceback
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionFinancialForecaster:
    """
    Production-ready financial forecasting system with comprehensive error handling.
    """
    
    def __init__(self):
        self.model_data = None
        self.is_loaded = False
        self.model_path = 'lightgbm_single_quarter_models.pkl'
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model with error handling."""
        try:
            if not Path(self.model_path).exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully. R² scores: {self.model_data.get('r2_scores', {})}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def validate_inputs(self, current_arr: float, net_new_arr: float, 
                       company_name: str = None) -> Tuple[bool, str]:
        """Comprehensive input validation."""
        try:
            # Basic validation
            if not isinstance(current_arr, (int, float)) or current_arr <= 0:
                return False, "Current ARR must be a positive number"
            
            if not isinstance(net_new_arr, (int, float)) or net_new_arr < 0:
                return False, "Net New ARR must be a non-negative number"
            
            if current_arr < net_new_arr:
                return False, "Net New ARR cannot be greater than Current ARR"
            
            # Growth rate validation
            growth_rate = (net_new_arr / (current_arr - net_new_arr)) * 100 if (current_arr - net_new_arr) > 0 else 0
            if growth_rate > 1000:  # 1000% YoY growth
                return False, "Growth rate too high (>1000%). Please verify inputs"
            
            # Size validation
            if current_arr > 1e12:  # $1 trillion
                return False, "ARR too large (>$1T). Please verify inputs"
            
            # Company name validation
            if company_name and len(company_name) > 100:
                return False, "Company name too long (>100 characters)"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def predict_with_fallback(self, current_arr: float, net_new_arr: float, 
                            company_name: str = "Anonymous Company", 
                            historical_arr: Dict = None) -> Dict[str, Any]:
        """
        Make prediction with comprehensive error handling and fallback.
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            is_valid, error_msg = self.validate_inputs(current_arr, net_new_arr, company_name)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "model_used": "None",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Check if model is loaded
            if not self.is_loaded:
                logger.warning("Model not loaded, using fallback calculation")
                return self._fallback_prediction(current_arr, net_new_arr, company_name, start_time)
            
            # Create test data - always use historical data approach
            from tests.test_final_solution import create_test_company_data_with_historical
            
            # Convert historical_arr dict to list if provided
            historical_list = None
            if historical_arr:
                historical_list = [
                    historical_arr.get('q1_arr'),
                    historical_arr.get('q2_arr'), 
                    historical_arr.get('q3_arr'),
                    historical_arr.get('q4_arr')
                ]
                # Filter out None values and ensure we have at least 4 quarters
                historical_list = [x for x in historical_list if x is not None]
                if len(historical_list) < 4:
                    historical_list = None
            
            # Always use the historical data function (it will generate realistic historical data if none provided)
            company_data = create_test_company_data_with_historical(
                name=company_name,
                arr_q4=current_arr,
                net_new_arr_q4=net_new_arr,
                headcount=max(1, int(current_arr / 200000)),  # Estimate headcount
                gross_margin=80.0,  # Default gross margin
                historical_arr=historical_list
            )
            
            # Make prediction
            from tests.test_final_solution import predict_with_documented_bias
            forecast_results = predict_with_documented_bias(
                self.model_data, 
                company_data, 
                company_name
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response
            response = {
                "success": True,
                "company_name": company_name,
                "input_metrics": {
                    "current_arr": current_arr,
                    "net_new_arr": net_new_arr,
                    "growth_rate": (net_new_arr / (current_arr - net_new_arr)) * 100 if (current_arr - net_new_arr) > 0 else 0
                },
                "forecast_results": forecast_results.to_dict('records'),
                "model_used": f"High-Accuracy Single-Quarter Models (R² = {np.mean(list(self.model_data['r2_scores'].values())):.4f})",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_limitations": {
                    "q1_bias": "Q1 predictions tend to be 7-8% optimistic",
                    "uncertainty": "±10% confidence intervals provided",
                    "data_requirements": "Model trained on VC-backed companies"
                }
            }
            
            logger.info(f"Prediction successful for {company_name} in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try fallback
            try:
                return self._fallback_prediction(current_arr, net_new_arr, company_name, start_time)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return {
                    "success": False,
                    "error": f"Prediction failed: {str(e)}",
                    "model_used": "None",
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
    
    def _fallback_prediction(self, current_arr: float, net_new_arr: float, 
                           company_name: str, start_time: datetime) -> Dict[str, Any]:
        """Fallback prediction when model is not available."""
        try:
            # Simple growth calculation
            growth_rate = (net_new_arr / (current_arr - net_new_arr)) * 100 if (current_arr - net_new_arr) > 0 else 0
            quarterly_growth = ((1 + growth_rate/100) ** (1/4) - 1) * 100
            
            # Generate 4 quarters of predictions
            forecast_results = []
            current_arr_value = current_arr
            
            for i in range(4):
                # Apply quarterly growth
                predicted_arr = current_arr_value * (1 + quarterly_growth/100)
                
                # Add uncertainty bounds
                uncertainty_factor = 0.15  # ±15% for fallback
                lower_bound = predicted_arr * (1 - uncertainty_factor)
                upper_bound = predicted_arr * (1 + uncertainty_factor)
                
                forecast_results.append({
                    "Future Quarter": f"FY24 Q{i+1}",
                    "Predicted ARR ($)": predicted_arr,
                    "Lower Bound ($)": lower_bound,
                    "Upper Bound ($)": upper_bound,
                    "Quarterly Growth (%)": quarterly_growth,
                    "YoY Growth (%)": growth_rate
                })
                
                current_arr_value = predicted_arr
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "company_name": company_name,
                "input_metrics": {
                    "current_arr": current_arr,
                    "net_new_arr": net_new_arr,
                    "growth_rate": growth_rate
                },
                "forecast_results": forecast_results,
                "model_used": "Fallback Calculation (Model not available)",
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "model_limitations": {
                    "fallback": "Using simple growth calculation",
                    "uncertainty": "±15% confidence intervals (higher uncertainty)",
                    "accuracy": "Lower accuracy than trained model"
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {str(e)}")
            return {
                "success": False,
                "error": f"Fallback calculation failed: {str(e)}",
                "model_used": "None",
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information."""
        return {
            "model_loaded": self.is_loaded,
            "model_path": self.model_path,
            "model_performance": self.model_data.get('r2_scores', {}) if self.model_data else {},
            "overall_r2": np.mean(list(self.model_data['r2_scores'].values())) if self.model_data and 'r2_scores' in self.model_data else 0,
            "timestamp": datetime.now().isoformat(),
            "system_health": "Healthy" if self.is_loaded else "Degraded"
        }
    
    def batch_predict(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple predictions in batch."""
        results = []
        
        for i, pred_request in enumerate(predictions):
            try:
                current_arr = pred_request.get('current_arr')
                net_new_arr = pred_request.get('net_new_arr')
                company_name = pred_request.get('company_name', f'Company_{i+1}')
                
                result = self.predict_with_fallback(current_arr, net_new_arr, company_name)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch prediction {i+1} failed: {str(e)}")
                results.append({
                    "success": False,
                    "error": f"Batch prediction failed: {str(e)}",
                    "model_used": "None"
                })
        
        return results

# Global instance for the API
production_forecaster = ProductionFinancialForecaster()

def get_production_forecaster() -> ProductionFinancialForecaster:
    """Get the global production forecaster instance."""
    return production_forecaster

# Example usage and testing
if __name__ == "__main__":
    # Test the production system
    forecaster = ProductionFinancialForecaster()
    
    # Test system status
    status = forecaster.get_system_status()
    print("System Status:", json.dumps(status, indent=2))
    
    # Test prediction
    result = forecaster.predict_with_fallback(
        current_arr=5000000,
        net_new_arr=1000000,
        company_name="Test Company"
    )
    
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2, default=str))


