#!/usr/bin/env python3
"""
Simple Model Predictions
=======================

Uses the model's predictions directly with minimal post-processing
to handle the drop issue. No hardcoded constraints.
"""

import pandas as pd
import numpy as np
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

def get_simple_predictions(company_data):
    """Get predictions using the model directly with minimal post-processing."""
    print("SIMPLE MODEL PREDICTIONS")
    print("=" * 50)
    
    # Get the last known quarter data
    last_quarter = company_data.iloc[-1]
    last_arr = last_quarter['ARR_End_of_Quarter']
    last_quarter_name = last_quarter['Quarter']
    
    print(f"Starting point: {last_quarter_name} = ${last_arr:,.0f}")
    
    # Get model predictions
    completion_system = IntelligentFeatureCompletionSystem()
    yoy_predictions, similar_companies, feature_vector = completion_system.predict_with_completed_features(company_data)
    
    print(f"\nModel's YoY Growth Predictions:")
    for i, pred in enumerate(yoy_predictions):
        print(f"  Q{i+1} 2024: {pred*100:.1f}% YoY growth")
    
    # Get the 2023 ARR values for YoY comparison
    q1_2023_arr = company_data.iloc[0]['ARR_End_of_Quarter']
    q2_2023_arr = company_data.iloc[1]['ARR_End_of_Quarter']
    q3_2023_arr = company_data.iloc[2]['ARR_End_of_Quarter']
    q4_2023_arr = company_data.iloc[3]['ARR_End_of_Quarter']
    
    # Calculate what the model's YoY predictions mean for 2024 ARR
    yoy_targets = []
    base_arrs = [q1_2023_arr, q2_2023_arr, q3_2023_arr, q4_2023_arr]
    
    print(f"\nModel's YoY Targets for 2024:")
    for i, (yoy_growth, base_arr) in enumerate(zip(yoy_predictions, base_arrs)):
        target_arr = base_arr * (1 + yoy_growth)
        yoy_targets.append(target_arr)
        print(f"  Q{i+1} 2024: ${target_arr:,.0f} (vs Q{i+1} 2023: ${base_arr:,.0f})")
    
    # Simple approach: Use the model's predictions directly
    # If there's a drop issue, just note it and let the user decide
    print(f"\nDIRECT MODEL PREDICTIONS:")
    print("-" * 40)
    
    predictions = []
    for i, (quarter, target_arr, yoy_growth) in enumerate(zip(
        ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
        yoy_targets,
        yoy_predictions
    )):
        predictions.append({
            'Quarter': quarter,
            'ARR': target_arr,
            'YoY_Growth': yoy_growth,
            'YoY_Growth_Percent': yoy_growth * 100
        })
        
        print(f"{quarter}: ${target_arr:,.0f} ({yoy_growth*100:.1f}% YoY growth)")
    
    # Check for the drop issue
    q1_2024_arr = yoy_targets[0]
    if q1_2024_arr < last_arr:
        print(f"\n⚠️  NOTE: Q1 2024 prediction (${q1_2024_arr:,.0f}) is lower than Q4 2023 (${last_arr:,.0f})")
        print(f"This is what the model predicts based on YoY growth calculations.")
        print(f"The model is comparing Q1 2024 to Q1 2023, not to Q4 2023.")
    
    return predictions

def main():
    """Test with the original company data."""
    # Load the original test company data
    company_df = pd.read_csv('test_company_2024.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_2024'
    
    # Calculate growth rates
    company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
    company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
    
    # Get simple predictions
    predictions = get_simple_predictions(company_df)
    
    print(f"\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print("Direct model predictions (no hardcoded constraints):")
    print()
    
    for pred in predictions:
        print(f"{pred['Quarter']}: ${pred['ARR']:,.0f} ({pred['YoY_Growth_Percent']:.1f}% YoY growth)")

if __name__ == "__main__":
    main()

