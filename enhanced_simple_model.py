#!/usr/bin/env python3
"""
Enhanced Simple Model with Better Feature Engineering
====================================================

Improves the simple model predictions through enhanced feature engineering
and smarter feature completion for extreme growth scenarios.
"""

import pandas as pd
import numpy as np
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

class EnhancedSimpleModel:
    def __init__(self):
        self.completion_system = IntelligentFeatureCompletionSystem()
        
    def enhance_company_data(self, company_df):
        """Enhance company data with better feature engineering."""
        print("Enhancing company data with advanced features...")
        
        # Calculate advanced growth features
        company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
        company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
        
        # Growth momentum features
        company_df['growth_momentum'] = company_df['qoq_growth'].rolling(2, min_periods=1).mean()
        company_df['growth_acceleration'] = company_df['qoq_growth'].diff()
        
        # Growth trend features
        company_df['growth_trend'] = company_df['qoq_growth'].rolling(3, min_periods=1).mean()
        company_df['is_accelerating'] = (company_df['growth_acceleration'] > 0).astype(int)
        
        # Company size and maturity features
        company_df['arr_size_small'] = (company_df['cARR'] < 1000000).astype(int)
        company_df['arr_size_medium'] = ((company_df['cARR'] >= 1000000) & (company_df['cARR'] < 10000000)).astype(int)
        company_df['arr_size_large'] = (company_df['cARR'] >= 10000000).astype(int)
        
        # High growth indicators
        company_df['is_high_growth'] = (company_df['qoq_growth'] > 0.2).astype(int)  # >20% QoQ growth
        company_df['is_hyper_growth'] = (company_df['qoq_growth'] > 0.5).astype(int)  # >50% QoQ growth
        
        # Efficiency features
        if 'Headcount (HC)' in company_df.columns:
            company_df['arr_per_employee'] = company_df['cARR'] / company_df['Headcount (HC)']
            company_df['efficiency_trend'] = company_df['arr_per_employee'].pct_change()
        
        # Market position features
        company_df['market_position'] = company_df['cARR'].rank(pct=True)
        
        # Growth stage features
        company_df['growth_stage_early'] = (company_df['cARR'] < 2000000).astype(int)
        company_df['growth_stage_scale'] = ((company_df['cARR'] >= 2000000) & (company_df['cARR'] < 10000000)).astype(int)
        company_df['growth_stage_mature'] = (company_df['cARR'] >= 10000000).astype(int)
        
        print(f"✅ Enhanced with {len([col for col in company_df.columns if col not in ['Quarter', 'ARR_End_of_Quarter', 'Headcount', 'Gross_Margin_Percent', 'Net_Profit_Loss_Margin_Percent']])} new features")
        
        return company_df
    
    def get_enhanced_predictions(self, company_df):
        """Get enhanced predictions using improved feature engineering."""
        print("ENHANCED SIMPLE MODEL PREDICTIONS")
        print("=" * 60)
        
        # Enhance the company data
        enhanced_df = self.enhance_company_data(company_df)
        
        # Get the last known quarter data
        last_quarter = enhanced_df.iloc[-1]
        last_arr = last_quarter['ARR_End_of_Quarter']
        last_quarter_name = last_quarter['Quarter']
        
        print(f"Starting point: {last_quarter_name} = ${last_arr:,.0f}")
        
        # Analyze growth patterns
        print(f"\nGrowth Pattern Analysis:")
        print(f"  Latest QoQ Growth: {last_quarter['qoq_growth']*100:.1f}%")
        print(f"  Growth Momentum: {last_quarter['growth_momentum']*100:.1f}%")
        print(f"  Growth Acceleration: {last_quarter['growth_acceleration']*100:.1f}%")
        print(f"  Is High Growth: {'Yes' if last_quarter['is_high_growth'] else 'No'}")
        print(f"  Is Hyper Growth: {'Yes' if last_quarter['is_hyper_growth'] else 'No'}")
        
        # Get model predictions with enhanced features
        yoy_predictions, similar_companies, feature_vector = self.completion_system.predict_with_completed_features(enhanced_df)
        
        print(f"\nModel's YoY Growth Predictions:")
        for i, pred in enumerate(yoy_predictions):
            print(f"  Q{i+1} 2024: {pred*100:.1f}% YoY growth")
        
        # Get the 2023 ARR values for YoY comparison
        q1_2023_arr = enhanced_df.iloc[0]['ARR_End_of_Quarter']
        q2_2023_arr = enhanced_df.iloc[1]['ARR_End_of_Quarter']
        q3_2023_arr = enhanced_df.iloc[2]['ARR_End_of_Quarter']
        q4_2023_arr = enhanced_df.iloc[3]['ARR_End_of_Quarter']
        
        # Calculate what the model's YoY predictions mean for 2024 ARR
        yoy_targets = []
        base_arrs = [q1_2023_arr, q2_2023_arr, q3_2023_arr, q4_2023_arr]
        
        print(f"\nModel's YoY Targets for 2024:")
        for i, (yoy_growth, base_arr) in enumerate(zip(yoy_predictions, base_arrs)):
            target_arr = base_arr * (1 + yoy_growth)
            yoy_targets.append(target_arr)
            print(f"  Q{i+1} 2024: ${target_arr:,.0f} (vs Q{i+1} 2023: ${base_arr:,.0f})")
        
        # Apply growth pattern adjustments
        print(f"\nApplying Growth Pattern Adjustments...")
        
        # If company is in hyper-growth mode, boost predictions
        if last_quarter['is_hyper_growth']:
            print(f"  🚀 Hyper-growth detected! Boosting predictions by 20%")
            yoy_targets = [arr * 1.2 for arr in yoy_targets]
        
        # If growth is accelerating, boost later quarters more
        if last_quarter['growth_acceleration'] > 0.1:  # >10% acceleration
            print(f"  📈 Growth acceleration detected! Boosting Q3-Q4 by 15%")
            yoy_targets[2] *= 1.15  # Q3
            yoy_targets[3] *= 1.15  # Q4
        
        # If company is in early growth stage, boost all predictions
        if last_quarter['growth_stage_early']:
            print(f"  🌱 Early growth stage detected! Boosting all predictions by 10%")
            yoy_targets = [arr * 1.1 for arr in yoy_targets]
        
        # Display final predictions with confidence intervals
        print(f"\nENHANCED PREDICTIONS WITH CONFIDENCE INTERVALS:")
        print("-" * 70)
        
        predictions = []
        for i, (quarter, target_arr, yoy_growth) in enumerate(zip(
            ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            yoy_targets,
            yoy_predictions
        )):
            # Calculate confidence intervals (±10%)
            pessimistic_arr = target_arr * 0.9  # -10%
            optimistic_arr = target_arr * 1.1   # +10%
            
            predictions.append({
                'Quarter': quarter,
                'ARR': target_arr,
                'Pessimistic_ARR': pessimistic_arr,
                'Optimistic_ARR': optimistic_arr,
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100
            })
            
            print(f"{quarter}: ${target_arr:,.0f} (${pessimistic_arr:,.0f} - ${optimistic_arr:,.0f}) ({yoy_growth*100:.1f}% YoY growth)")
        
        # Check for the drop issue
        q1_2024_arr = yoy_targets[0]
        if q1_2024_arr < last_arr:
            print(f"\n⚠️  NOTE: Q1 2024 prediction (${q1_2024_arr:,.0f}) is lower than Q4 2023 (${last_arr:,.0f})")
            print(f"This is what the model predicts based on YoY growth calculations.")
            print(f"The model is comparing Q1 2024 to Q1 2023, not to Q4 2023.")
        
        return predictions, enhanced_df

def test_enhanced_model():
    """Test the enhanced simple model."""
    # Load the original test company data
    company_df = pd.read_csv('test_company_2024.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_2024'
    
    # Initialize enhanced model
    enhanced_model = EnhancedSimpleModel()
    
    # Get enhanced predictions
    predictions, enhanced_df = enhanced_model.get_enhanced_predictions(company_df)
    
    print(f"\n" + "=" * 60)
    print("FINAL ENHANCED RESULTS WITH CONFIDENCE INTERVALS")
    print("=" * 60)
    print("Enhanced model predictions with ±10% confidence intervals:")
    print()
    
    for pred in predictions:
        print(f"{pred['Quarter']}: ${pred['ARR']:,.0f} (${pred['Pessimistic_ARR']:,.0f} - ${pred['Optimistic_ARR']:,.0f}) ({pred['YoY_Growth_Percent']:.1f}% YoY growth)")
    
    # Compare with actual results
    actual_results = {
        'Q1 2024': 3800000,
        'Q2 2024': 4900000,
        'Q3 2024': 6100000,
        'Q4 2024': 7500000
    }
    
    print(f"\n" + "=" * 60)
    print("COMPARISON WITH ACTUAL 2024 RESULTS")
    print("=" * 60)
    
    print(f"{'Quarter':<12} {'Predicted':<15} {'Range':<25} {'Actual':<15} {'Error':<12} {'Error %':<10}")
    print("-" * 90)
    
    total_error = 0
    for pred in predictions:
        quarter = pred['Quarter']
        predicted = pred['ARR']
        pessimistic = pred['Pessimistic_ARR']
        optimistic = pred['Optimistic_ARR']
        actual = actual_results[quarter]
        error = actual - predicted
        error_pct = (error / actual) * 100 if actual != 0 else 0
        
        range_str = f"${pessimistic:,.0f}-${optimistic:,.0f}"
        print(f"{quarter:<12} ${predicted:<14,.0f} {range_str:<25} ${actual:<14,.0f} ${error:<11,.0f} {error_pct:<9.1f}%")
        total_error += abs(error)
    
    avg_error = total_error / 4
    print(f"\nAverage Absolute Error: ${avg_error:,.0f}")
    
    return predictions, enhanced_df

if __name__ == "__main__":
    predictions, enhanced_df = test_enhanced_model()

