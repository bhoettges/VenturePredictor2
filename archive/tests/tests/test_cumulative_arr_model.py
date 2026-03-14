#!/usr/bin/env python3
"""
Test Cumulative ARR Model
========================

Test the new cumulative ARR prediction model with real company data.
"""

import pandas as pd
import numpy as np
import joblib
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

def test_cumulative_arr_model():
    """Test the cumulative ARR model with the new company data."""
    print("TESTING CUMULATIVE ARR MODEL")
    print("=" * 50)
    
    # Load the new company data
    company_df = pd.read_csv('test_company_new.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_new'
    
    # Calculate cumulative ARR
    company_df['Cumulative_ARR'] = company_df['cARR'].cumsum()
    
    print("Company Data:")
    for i, row in company_df.iterrows():
        print(f"  {row['Quarter']}: ARR=${row['cARR']:,.0f}, Cumulative=${row['Cumulative_ARR']:,.0f}")
    
    # Get the last known cumulative ARR
    last_cumulative_arr = company_df['Cumulative_ARR'].iloc[-1]
    print(f"\nStarting Cumulative ARR (Q4 2023): ${last_cumulative_arr:,.0f}")
    
    # Load the cumulative ARR model
    print(f"\nLoading cumulative ARR model...")
    model_data = joblib.load('cumulative_arr_model.pkl')
    
    # Prepare features for the model
    feature_cols = model_data['feature_cols']
    
    # Create a feature vector for the last quarter
    last_quarter = company_df.iloc[-1]
    feature_vector = []
    
    for col in feature_cols:
        if col in last_quarter:
            feature_vector.append(last_quarter[col])
        else:
            feature_vector.append(0)  # Default value
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    
    # Scale and select features
    scaler = model_data['scaler']
    feature_selector = model_data['feature_selector']
    
    X_scaled = scaler.transform(feature_vector)
    X_selected = feature_selector.transform(X_scaled)
    
    # Make predictions
    models = model_data['models']
    target_cols = model_data['target_cols']
    
    print(f"\nMaking predictions...")
    predictions = {}
    
    for target_col in target_cols:
        pred = models[target_col].predict(X_selected)[0]
        predictions[target_col] = pred
        print(f"  {target_col}: {pred:.3f} ({pred*100:.1f}% growth)")
    
    # Calculate cumulative ARR for 2024
    print(f"\n" + "=" * 50)
    print("CUMULATIVE ARR PREDICTIONS FOR 2024")
    print("=" * 50)
    
    cumulative_arr_2024 = {}
    cumulative_arr_2024['Q1'] = last_cumulative_arr * (1 + predictions['Cumulative_ARR_Growth_Q1'])
    cumulative_arr_2024['Q2'] = last_cumulative_arr * (1 + predictions['Cumulative_ARR_Growth_Q2'])
    cumulative_arr_2024['Q3'] = last_cumulative_arr * (1 + predictions['Cumulative_ARR_Growth_Q3'])
    cumulative_arr_2024['Q4'] = last_cumulative_arr * (1 + predictions['Cumulative_ARR_Growth_Q4'])
    
    print(f"Starting point (Q4 2023): ${last_cumulative_arr:,.0f}")
    print()
    
    for quarter, cumulative_arr in cumulative_arr_2024.items():
        growth = predictions[f'Cumulative_ARR_Growth_{quarter}']
        print(f"Q{quarter[-1]} 2024: ${cumulative_arr:,.0f} ({growth*100:.1f}% cumulative growth)")
    
    # Calculate quarterly ARR contributions
    print(f"\n" + "=" * 50)
    print("QUARTERLY ARR BREAKDOWN")
    print("=" * 50)
    
    quarterly_arr = {}
    quarterly_arr['Q1'] = cumulative_arr_2024['Q1'] - last_cumulative_arr
    quarterly_arr['Q2'] = cumulative_arr_2024['Q2'] - cumulative_arr_2024['Q1']
    quarterly_arr['Q3'] = cumulative_arr_2024['Q3'] - cumulative_arr_2024['Q2']
    quarterly_arr['Q4'] = cumulative_arr_2024['Q4'] - cumulative_arr_2024['Q3']
    
    print(f"Quarterly ARR Contributions:")
    for quarter, arr in quarterly_arr.items():
        print(f"  Q{quarter[-1]} 2024: ${arr:,.0f}")
    
    # Summary
    total_2024_arr = cumulative_arr_2024['Q4']
    total_growth = (total_2024_arr - last_cumulative_arr) / last_cumulative_arr * 100
    
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Starting Cumulative ARR (Q4 2023): ${last_cumulative_arr:,.0f}")
    print(f"Ending Cumulative ARR (Q4 2024): ${total_2024_arr:,.0f}")
    print(f"Total Growth: {total_growth:.1f}%")
    print(f"Average Quarterly Contribution: ${np.mean(list(quarterly_arr.values())):,.0f}")
    
    return {
        'cumulative_arr_2024': cumulative_arr_2024,
        'quarterly_arr': quarterly_arr,
        'predictions': predictions
    }

if __name__ == "__main__":
    results = test_cumulative_arr_model()

