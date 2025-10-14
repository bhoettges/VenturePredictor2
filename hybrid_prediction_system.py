#!/usr/bin/env python3
"""
Hybrid Prediction System
========================

Combines ML model (for growth) and GPT (for edge cases) using multi-factor trend detection.

Flow:
1. Detect company trend using multi-factor analysis
2. Route to ML model (93% of cases) or GPT (edge cases)
3. Return predictions with reasoning
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem
from trend_detector import TrendDetector
from gpt_predictor import GPTPredictor

class HybridPredictionSystem:
    """Hybrid system that uses ML for growth and GPT for edge cases."""
    
    def __init__(self):
        self.ml_system = IntelligentFeatureCompletionSystem()
        self.trend_detector = TrendDetector()
        try:
            self.gpt_predictor = GPTPredictor()
            self.gpt_available = True
        except ValueError as e:
            print(f"⚠️ GPT predictor not available: {e}")
            print("   Will fall back to ML model for all predictions")
            self.gpt_available = False
    
    def create_company_dataframe(self, tier1_data: Dict, tier2_data: Optional[Dict] = None) -> pd.DataFrame:
        """Create company dataframe from tier-based input."""
        
        # Tier 1 data (required)
        quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
        arr_values = [tier1_data['q1_arr'], tier1_data['q2_arr'], 
                     tier1_data['q3_arr'], tier1_data['q4_arr']]
        
        # Create base dataframe
        company_df = pd.DataFrame({
            'Quarter': quarters,
            'ARR_End_of_Quarter': arr_values,
            'Financial Quarter': quarters,
            'cARR': arr_values,
            'Headcount (HC)': [tier1_data['headcount']] * 4,
            'Sector': [tier1_data['sector']] * 4,
            'id_company': 'user_company'
        })
        
        # Calculate growth rates
        company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
        company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
        
        # Add Tier 2 data if provided
        if tier2_data:
            company_df['Gross Margin (in %)'] = tier2_data.get('gross_margin', 75)
            company_df['Sales & Marketing'] = tier2_data.get('sales_marketing', company_df['cARR'].iloc[-1] * 0.4)
            company_df['Cash Burn (OCF & ICF)'] = tier2_data.get('cash_burn', -company_df['cARR'].iloc[-1] * 0.3)
            company_df['Customers (EoP)'] = tier2_data.get('customers', int(company_df['cARR'].iloc[-1] / 5000))
            
            churn_rate = tier2_data.get('churn_rate', 0.05)
            expansion_rate = tier2_data.get('expansion_rate', 0.10)
            
            company_df['Churn & Reduction'] = -company_df['cARR'] * churn_rate
            company_df['Expansion & Upsell'] = company_df['cARR'] * expansion_rate
        else:
            # Use intelligent defaults
            last_arr = company_df['cARR'].iloc[-1]
            company_df['Gross Margin (in %)'] = 75
            company_df['Sales & Marketing'] = last_arr * 0.4
            company_df['Cash Burn (OCF & ICF)'] = -last_arr * 0.3
            company_df['Customers (EoP)'] = int(last_arr / 5000)
            company_df['Churn & Reduction'] = -last_arr * 0.05
            company_df['Expansion & Upsell'] = last_arr * 0.10
        
        return company_df
    
    def predict_with_hybrid(self, tier1_data: Dict, tier2_data: Optional[Dict] = None) -> Tuple[list, Dict]:
        """
        Make predictions using hybrid ML/GPT approach.
        
        Returns:
            Tuple of (predictions list, metadata dict with trend analysis)
        """
        
        print("\n" + "=" * 80)
        print("🚀 HYBRID PREDICTION SYSTEM (ML + GPT)")
        print("=" * 80)
        
        # Extract ARR values
        q1 = tier1_data['q1_arr']
        q2 = tier1_data['q2_arr']
        q3 = tier1_data['q3_arr']
        q4 = tier1_data['q4_arr']
        
        # Step 1: Detect trend
        print("\n📊 STEP 1: TREND ANALYSIS")
        print("-" * 80)
        trend_analysis = self.trend_detector.detect_trend(q1, q2, q3, q4)
        
        print(f"Trend Type: {trend_analysis['trend_type']}")
        print(f"Confidence: {trend_analysis['confidence']}")
        print(f"Reason: {trend_analysis['reason']}")
        print(f"\n{trend_analysis['user_message']}")
        
        # Step 2: Route to appropriate prediction method
        print("\n🔮 STEP 2: PREDICTION")
        print("-" * 80)
        
        if trend_analysis['use_gpt'] and self.gpt_available:
            print("✅ Using GPT for contextual prediction (edge case detected)")
            
            # Use GPT
            result = self.gpt_predictor.predict_arr(
                q1=q1, q2=q2, q3=q3, q4=q4,
                sector=tier1_data['sector'],
                headcount=tier1_data['headcount'],
                trend_analysis=trend_analysis
            )
            
            if result['success']:
                predictions = result['predictions']
                metadata = {
                    'prediction_method': 'GPT',
                    'trend_analysis': trend_analysis,
                    'gpt_reasoning': result['gpt_reasoning'],
                    'gpt_confidence': result['gpt_confidence'],
                    'gpt_assumption': result.get('gpt_assumption', 'N/A'),
                    'fallback_used': result.get('fallback_used', False)
                }
                
                print(f"\n💡 GPT Reasoning: {result['gpt_reasoning']}")
                print(f"🎯 Confidence: {result['gpt_confidence']}")
                
            else:
                # GPT failed, fall back to ML
                print("⚠️ GPT prediction failed, falling back to ML model")
                predictions, metadata = self._use_ml_model(tier1_data, tier2_data, trend_analysis)
        
        else:
            # Use ML model
            if not trend_analysis['use_gpt']:
                print("✅ Using ML model (standard growth pattern)")
            else:
                print("⚠️ GPT not available, using ML model as fallback")
            
            predictions, metadata = self._use_ml_model(tier1_data, tier2_data, trend_analysis)
        
        # Step 3: Calculate QoQ growth for all predictions
        current_arr = q4
        for pred in predictions:
            qoq_growth = ((pred['ARR'] - current_arr) / current_arr) * 100 if current_arr > 0 else 0
            pred['QoQ_Growth_Percent'] = qoq_growth
            current_arr = pred['ARR']
        
        # Display results
        print("\n📈 PREDICTIONS:")
        print("-" * 80)
        for pred in predictions:
            print(f"{pred['Quarter']}: ${pred['ARR']:,.0f} "
                  f"(YoY: {pred['YoY_Growth_Percent']:+.1f}%, "
                  f"QoQ: {pred['QoQ_Growth_Percent']:+.1f}%)")
        
        return predictions, metadata
    
    def _use_ml_model(self, tier1_data: Dict, tier2_data: Optional[Dict], trend_analysis: Dict) -> Tuple[list, Dict]:
        """Use ML model for predictions."""
        
        # Create company dataframe
        company_df = self.create_company_dataframe(tier1_data, tier2_data)
        
        # Get ML predictions
        yoy_predictions, similar_companies, feature_vector = self.ml_system.predict_with_completed_features(company_df)
        
        # Format predictions
        predictions = []
        arr_values = [tier1_data['q1_arr'], tier1_data['q2_arr'], 
                     tier1_data['q3_arr'], tier1_data['q4_arr']]
        
        for i, (quarter, yoy_growth) in enumerate(zip(
            ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            yoy_predictions
        )):
            # YoY prediction relative to same quarter last year
            base_quarter_arr = arr_values[i]
            target_arr = base_quarter_arr * yoy_growth
            
            predictions.append({
                'Quarter': quarter,
                'ARR': target_arr,
                'Pessimistic_ARR': target_arr * 0.9,
                'Optimistic_ARR': target_arr * 1.1,
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100,
                'QoQ_Growth_Percent': 0  # Will be calculated later
            })
        
        metadata = {
            'prediction_method': 'ML_Model',
            'trend_analysis': trend_analysis,
            'model_accuracy': 'R² = 0.7966 (79.66%)',
            'similar_companies_found': len(similar_companies)
        }
        
        return predictions, metadata

def test_hybrid_system():
    """Test the hybrid system with different scenarios."""
    
    system = HybridPredictionSystem()
    
    test_cases = [
        {
            'name': 'DECLINING COMPANY',
            'data': {
                'q1_arr': 2000000,
                'q2_arr': 1500000,
                'q3_arr': 1000000,
                'q4_arr': 500000,
                'headcount': 50,
                'sector': 'Data & Analytics'
            }
        },
        {
            'name': 'GROWING COMPANY',
            'data': {
                'q1_arr': 1000000,
                'q2_arr': 1400000,
                'q3_arr': 2000000,
                'q4_arr': 2800000,
                'headcount': 100,
                'sector': 'Data & Analytics'
            }
        },
        {
            'name': 'V-SHAPE RECOVERY',
            'data': {
                'q1_arr': 2000000,
                'q2_arr': 1000000,
                'q3_arr': 1200000,
                'q4_arr': 1800000,
                'headcount': 75,
                'sector': 'Cyber Security'
            }
        }
    ]
    
    for test_case in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST CASE: {test_case['name']}")
        print("=" * 80)
        
        predictions, metadata = system.predict_with_hybrid(test_case['data'])
        
        print(f"\n✅ Prediction Method: {metadata['prediction_method']}")
        print(f"✅ Trend Type: {metadata['trend_analysis']['trend_type']}")
        
        if 'gpt_reasoning' in metadata:
            print(f"\n💡 GPT Reasoning: {metadata['gpt_reasoning']}")

if __name__ == "__main__":
    test_hybrid_system()

