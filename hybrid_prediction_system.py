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
from rule_based_health_predictor import RuleBasedHealthPredictor

class HybridPredictionSystem:
    """Hybrid system that uses ML for growth and rule-based health assessment for edge cases."""
    
    def __init__(self):
        self.ml_system = IntelligentFeatureCompletionSystem()
        self.trend_detector = TrendDetector()
        self.rule_based_predictor = RuleBasedHealthPredictor()
    
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
        
        # Calculate growth rates from actual data (pct_change(4) is NaN with only 4 rows)
        annual_growth = (arr_values[-1] - arr_values[0]) / arr_values[0] if arr_values[0] > 0 else 0
        company_df['yoy_growth'] = annual_growth
        company_df['ARR YoY Growth (in %)'] = annual_growth * 100
        company_df['Revenue YoY Growth (in %)'] = annual_growth * 100
        company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
        
        # --- Derivable features: Revenue & ARR (directly from Tier 1) ---
        q1, q2, q3, q4 = arr_values
        hc = tier1_data['headcount']
        net_new_per_q = [q2 - q1, q3 - q2, q4 - q3, q4 - q3]

        company_df['Revenue'] = arr_values
        company_df['LTM Revenue'] = [sum(arr_values)] * 4
        company_df['Revenue Run Rate (RRR)'] = arr_values
        company_df['Net New ARR'] = net_new_per_q
        company_df['ARR / HC'] = [a / hc if hc > 0 else 0 for a in arr_values]

        # --- Tier 2 data (provided or sensible defaults) ---
        t2 = tier2_data or {}
        gross_margin = t2.get('gross_margin', 75)
        sm = t2.get('sales_marketing', q4 * 0.4)
        cash_burn = t2.get('cash_burn', -q4 * 0.3)
        customers = t2.get('customers', max(1, int(q4 / 5000)))
        churn_rate = t2.get('churn_rate', 0.05)
        expansion_rate = t2.get('expansion_rate', 0.10)

        company_df['Gross Margin (in %)'] = [gross_margin] * 4
        company_df['Sales & Marketing'] = [sm] * 4
        company_df['Cash Burn (OCF & ICF)'] = [cash_burn] * 4
        company_df['Customers (EoP)'] = [customers] * 4
        company_df['Churn & Reduction'] = [-a * churn_rate for a in arr_values]
        company_df['Expansion & Upsell'] = [a * expansion_rate for a in arr_values]

        # --- Derivable efficiency metrics (from Tier 1 + Tier 2) ---
        company_df['Gross Profit'] = [a * gross_margin / 100 for a in arr_values]
        company_df['S&M as % of Revenue'] = [sm / q4 * 100 if q4 > 0 else 0] * 4
        magic = (net_new_per_q[-1] * 4) / sm if sm > 0 else 0
        company_df['LTM Magic Number (ARR)'] = [magic] * 4
        ebitda_margin = gross_margin - 35  # rough EBITDA estimate
        rule_of_40 = (annual_growth * 100) + ebitda_margin
        company_df['LTM Rule of 40% (ARR)'] = [rule_of_40] * 4

        # --- Lag and rolling features (must match training pipeline) ---
        # The model was trained with these temporal features; without them,
        # the feature completion system infers them from similar companies
        # and the model loses sight of this company's actual growth momentum.
        metrics_to_process = [
            'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
            'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
        ]
        for col in metrics_to_process:
            if col not in company_df.columns:
                continue
            for lag in [1, 2, 4]:
                company_df[f'{col}_lag_{lag}'] = company_df[col].shift(lag)
            company_df[f'{col}_roll_mean_4q'] = company_df[col].rolling(window=4, min_periods=1).mean().shift(1)
            company_df[f'{col}_roll_std_4q'] = company_df[col].rolling(window=4, min_periods=1).std().shift(1)

        sm_lag1 = company_df['Sales & Marketing'].shift(1)
        net_new = company_df['Net New ARR']
        company_df['Magic_Number'] = net_new / sm_lag1
        company_df['Burn_Multiple'] = company_df['Cash Burn (OCF & ICF)'].abs() / net_new
        company_df['HC_qoq_growth'] = company_df['Headcount (HC)'].pct_change(1)
        company_df['ARR_per_Headcount'] = company_df['cARR'] / company_df['Headcount (HC)']

        company_df = company_df.replace([np.inf, -np.inf], np.nan)

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

        # Optional: health-score routing override (best win)
        # If Tier 2 metrics indicate poor fundamentals (e.g., low health score or short runway),
        # we route to the rule-based health system even if ARR trend looks "normal".
        routing_override = {
            "applied": False,
            "reason": None,
            "health_tier": None,
            "health_score": None,
            "runway_months": None
        }
        tier2_has_any_signal = bool(tier2_data) and any(v is not None for v in (tier2_data or {}).values())
        if tier2_has_any_signal:
            try:
                health_metrics = self.rule_based_predictor.calculate_health_metrics(q1, q2, q3, q4, tier2_data)
                health_tier, health_assessment = self.rule_based_predictor.assess_health_tier(health_metrics)
                runway_months = health_metrics.get("runway_months", None)
                health_score = health_assessment.get("score", None)

                routing_override.update({
                    "health_tier": health_tier,
                    "health_score": health_score,
                    "runway_months": runway_months
                })

                # Override criteria:
                # - LOW health tier OR
                # - runway below minimum threshold (capital constraint risk)
                min_runway = getattr(self.rule_based_predictor, "MINIMUM_RUNWAY_MONTHS", 12)
                if health_tier == "LOW":
                    routing_override["applied"] = True
                    routing_override["reason"] = f"Health tier LOW (score {health_score}/100) from Tier 2 signals"
                elif runway_months is not None and runway_months < min_runway:
                    routing_override["applied"] = True
                    routing_override["reason"] = f"Short runway ({runway_months:.0f} months) below minimum ({min_runway})"
            except Exception:
                # If health scoring fails for any reason, don't block predictions
                pass
        
        # Step 2: Route to appropriate prediction method
        print("\n🔮 STEP 2: PREDICTION")
        print("-" * 80)
        
        if trend_analysis['use_gpt'] or routing_override["applied"]:
            print("✅ Using rule-based health assessment (edge case detected)")
            
            # Use rule-based health predictor
            result = self.rule_based_predictor.predict_arr(
                q1=q1, q2=q2, q3=q3, q4=q4,
                sector=tier1_data['sector'],
                headcount=tier1_data['headcount'],
                trend_analysis=trend_analysis,
                tier2_data=tier2_data
            )
            
            if result['success']:
                predictions = result['predictions']
                metadata = {
                    'prediction_method': 'Rule-Based Health Assessment',
                    'trend_analysis': trend_analysis,
                    'health_tier': result['health_tier'],
                    'health_assessment': result['health_assessment'],
                    'health_metrics': result['health_metrics'],
                    'reasoning': result['reasoning'],
                    'confidence': result['confidence'],
                    'key_assumption': result.get('key_assumption', 'N/A'),
                    'routing_override': routing_override
                }
                
                print(f"\n🏥 Health Tier: {result['health_tier']} (Score: {result['health_assessment']['score']}/100)")
                print(f"💡 Reasoning: {result['reasoning']}")
                print(f"🎯 Confidence: {result['confidence']}")
                
            else:
                # Rule-based failed, fall back to ML
                print("⚠️ Rule-based prediction failed, falling back to ML model")
                predictions, metadata = self._use_ml_model(tier1_data, tier2_data, trend_analysis)
        
        else:
            # Use ML model
            print("✅ Using ML model (standard growth pattern)")
            predictions, metadata = self._use_ml_model(tier1_data, tier2_data, trend_analysis)
            metadata['routing_override'] = routing_override
        
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
        
        # Get ML predictions (model outputs YoY growth in PERCENT, e.g. 54.4 = 54.4%)
        yoy_predictions_pct, similar_companies, feature_vector = self.ml_system.predict_with_completed_features(company_df)
        
        arr_values = [tier1_data['q1_arr'], tier1_data['q2_arr'], 
                     tier1_data['q3_arr'], tier1_data['q4_arr']]
        
        # The model predicts independent YoY growth for each quarter relative to
        # the same quarter last year.  Applying these directly creates a visual
        # "dip" because Q1_2024 = Q1_2023 * (1+yoy) can be lower than Q4_2023
        # for a growing company.  SaaS ARR is not seasonal, so we convert the
        # model's YoY signal into a sequential forward projection from Q4 2023.

        yoy_fracs = [pct / 100.0 for pct in yoy_predictions_pct]
        arr_yoy_implied = [base * (1 + yoy) for base, yoy in zip(arr_values, yoy_fracs)]

        total_2023 = sum(arr_values)
        total_2024_implied = sum(arr_yoy_implied)
        implied_annual_growth = (total_2024_implied / total_2023) - 1 if total_2023 > 0 else 0
        qoq_rate = (1 + implied_annual_growth) ** 0.25 - 1

        predictions = []
        current_arr = arr_values[-1]  # Q4 2023

        for i, quarter in enumerate(['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']):
            current_arr = current_arr * (1 + qoq_rate)
            base_quarter_arr = arr_values[i]
            yoy_growth = (current_arr - base_quarter_arr) / base_quarter_arr if base_quarter_arr > 0 else 0

            predictions.append({
                'Quarter': quarter,
                'ARR': current_arr,
                'Pessimistic_ARR': current_arr * 0.9,
                'Optimistic_ARR': current_arr * 1.1,
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100,
                'QoQ_Growth_Percent': qoq_rate * 100
            })
        
        metadata = {
            'prediction_method': 'ML_Model',
            'trend_analysis': trend_analysis,
            'model_accuracy': 'R² = 0.7966 (79.66%)',
            'similar_companies_found': len(similar_companies),
            'raw_yoy_predictions_pct': list(yoy_predictions_pct),
            'implied_annual_growth_pct': implied_annual_growth * 100,
            'implied_qoq_growth_pct': qoq_rate * 100
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
        
        if 'health_tier' in metadata:
            print(f"\n🏥 Health Tier: {metadata['health_tier']}")
            print(f"💡 Reasoning: {metadata['reasoning']}")

if __name__ == "__main__":
    test_hybrid_system()

