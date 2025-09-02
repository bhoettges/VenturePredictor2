"""
Prediction Memory System
Stores and manages recent predictions for chat analysis
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class PredictionRecord:
    """Single prediction record"""
    timestamp: str
    company_name: str
    prediction_type: str  # 'tier_based', 'csv', 'chat'
    input_data: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    insights: Dict[str, Any]
    model_used: str
    success: bool
    error: Optional[str] = None

class PredictionMemory:
    """Manages prediction history for chat analysis"""
    
    def __init__(self, max_records: int = 10):
        self.max_records = max_records
        self.memory_file = "prediction_memory.json"
        self.predictions: List[PredictionRecord] = []
        self.load_memory()
    
    def add_prediction(self, 
                      company_name: str,
                      prediction_type: str,
                      input_data: Dict[str, Any],
                      predictions: List[Dict[str, Any]],
                      insights: Dict[str, Any],
                      model_used: str,
                      success: bool = True,
                      error: Optional[str] = None):
        """Add a new prediction to memory"""
        
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            company_name=company_name,
            prediction_type=prediction_type,
            input_data=input_data,
            predictions=predictions,
            insights=insights,
            model_used=model_used,
            success=success,
            error=error
        )
        
        # Add to beginning of list (most recent first)
        self.predictions.insert(0, record)
        
        # Keep only the most recent records
        if len(self.predictions) > self.max_records:
            self.predictions = self.predictions[:self.max_records]
        
        self.save_memory()
        return record
    
    def get_recent_predictions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent predictions as dictionaries for chat analysis"""
        recent = self.predictions[:limit]
        return [asdict(record) for record in recent]
    
    def get_prediction_by_company(self, company_name: str) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction for a specific company"""
        for record in self.predictions:
            if record.company_name.lower() == company_name.lower():
                return asdict(record)
        return None
    
    def get_prediction_summary(self) -> str:
        """Get a summary of recent predictions for chat context"""
        if not self.predictions:
            return "No recent predictions available."
        
        summary = f"Recent Predictions ({len(self.predictions)} total):\n\n"
        
        for i, record in enumerate(self.predictions[:3], 1):
            status = "✅ Success" if record.success else "❌ Failed"
            summary += f"{i}. **{record.company_name}** ({record.prediction_type})\n"
            summary += f"   - {status} - {record.timestamp}\n"
            summary += f"   - Model: {record.model_used}\n"
            
            if record.success and record.predictions:
                # Show key prediction metrics
                if 'predicted_final_arr' in record.insights:
                    current_arr = record.insights.get('current_arr', 0)
                    final_arr = record.insights.get('predicted_final_arr', 0)
                    growth = record.insights.get('total_growth_percent', 0)
                    summary += f"   - Growth: {growth:.1f}% (${current_arr:,.0f} → ${final_arr:,.0f})\n"
            
            summary += "\n"
        
        return summary
    
    def save_memory(self):
        """Save predictions to file"""
        try:
            data = [asdict(record) for record in self.predictions]
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save prediction memory: {e}")
    
    def load_memory(self):
        """Load predictions from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                self.predictions = [PredictionRecord(**record) for record in data]
        except Exception as e:
            print(f"Warning: Could not load prediction memory: {e}")
            self.predictions = []

# Global instance
prediction_memory = PredictionMemory()

def add_tier_based_prediction(result: Dict[str, Any], input_data: Dict[str, Any]):
    """Add a tier-based prediction to memory"""
    if result.get("success"):
        prediction_memory.add_prediction(
            company_name=result.get("company_name", "Unknown"),
            prediction_type="tier_based",
            input_data=input_data,
            predictions=result.get("forecast", []),
            insights=result.get("insights", {}),
            model_used=result.get("model_used", "Unknown"),
            success=True
        )
    else:
        prediction_memory.add_prediction(
            company_name=result.get("company_name", "Unknown"),
            prediction_type="tier_based",
            input_data=input_data,
            predictions=[],
            insights={},
            model_used="Unknown",
            success=False,
            error=result.get("error", "Unknown error")
        )

def add_csv_prediction(result: Dict[str, Any], input_data: Dict[str, Any]):
    """Add a CSV prediction to memory"""
    if result.get("success"):
        prediction_memory.add_prediction(
            company_name=result.get("company_name", "CSV Upload"),
            prediction_type="csv",
            input_data=input_data,
            predictions=result.get("forecast", []),
            insights=result.get("insights", {}),
            model_used=result.get("model_used", "Unknown"),
            success=True
        )
    else:
        prediction_memory.add_prediction(
            company_name=result.get("company_name", "CSV Upload"),
            prediction_type="csv",
            input_data=input_data,
            predictions=[],
            insights={},
            model_used="Unknown",
            success=False,
            error=result.get("error", "Unknown error")
        )

