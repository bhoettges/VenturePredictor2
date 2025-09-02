#!/usr/bin/env python3
"""
Test New Company with Simple Predictions
=======================================

Test the clean model predictions with the new company data.
"""

import pandas as pd
import numpy as np
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

def test_new_company():
    """Test the new company data through the simple prediction system."""
    print("TESTING NEW COMPANY WITH SIMPLE PREDICTIONS")
    print("=" * 60)
    
    # Load the new company data
    company_df = pd.read_csv('test_company_new.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_new'
    
    # Calculate growth rates
    company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
    company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
    
    print("Company Data:")
    for i, row in company_df.iterrows():
        print(f"  {row['Quarter']}: ARR=${row['cARR']:,.0f}, Headcount={row['Headcount (HC)']}")
    
    print(f"\nGrowth Analysis:")
    for i, row in company_df.iterrows():
        if i > 0:
            qoq = row['qoq_growth'] * 100
            print(f"  {row['Quarter']}: QoQ Growth = {qoq:.1f}%")
    
    print(f"\nCurrent Status (Q4 2023):")
    print(f"  ARR: ${company_df['cARR'].iloc[-1]:,.0f}")
    print(f"  Headcount: {company_df['Headcount (HC)'].iloc[-1]}")
    print(f"  QoQ Growth: {company_df['qoq_growth'].iloc[-1]*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("GETTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Get the last known quarter data
    last_quarter = company_df.iloc[-1]
    last_arr = last_quarter['ARR_End_of_Quarter']
    last_quarter_name = last_quarter['Quarter']
    
    print(f"Starting point: {last_quarter_name} = ${last_arr:,.0f}")
    
    # Get model predictions
    completion_system = IntelligentFeatureCompletionSystem()
    yoy_predictions, similar_companies, feature_vector = completion_system.predict_with_completed_features(company_df)
    
    print(f"\nModel's YoY Growth Predictions:")
    for i, pred in enumerate(yoy_predictions):
        print(f"  Q{i+1} 2024: {pred*100:.1f}% YoY growth")
    
    # Get the 2023 ARR values for YoY comparison
    q1_2023_arr = company_df.iloc[0]['ARR_End_of_Quarter']
    q2_2023_arr = company_df.iloc[1]['ARR_End_of_Quarter']
    q3_2023_arr = company_df.iloc[2]['ARR_End_of_Quarter']
    q4_2023_arr = company_df.iloc[3]['ARR_End_of_Quarter']
    
    # Calculate what the model's YoY predictions mean for 2024 ARR
    yoy_targets = []
    base_arrs = [q1_2023_arr, q2_2023_arr, q3_2023_arr, q4_2023_arr]
    
    print(f"\nModel's YoY Targets for 2024:")
    for i, (yoy_growth, base_arr) in enumerate(zip(yoy_predictions, base_arrs)):
        target_arr = base_arr * (1 + yoy_growth)
        yoy_targets.append(target_arr)
        print(f"  Q{i+1} 2024: ${target_arr:,.0f} (vs Q{i+1} 2023: ${base_arr:,.0f})")
    
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Direct model predictions (no hardcoded constraints):")
    print()
    
    # Check for the drop issue
    q1_2024_arr = yoy_targets[0]
    if q1_2024_arr < last_arr:
        print(f"⚠️  NOTE: Q1 2024 prediction (${q1_2024_arr:,.0f}) is lower than Q4 2023 (${last_arr:,.0f})")
        print(f"This is what the model predicts based on YoY growth calculations.")
        print(f"The model is comparing Q1 2024 to Q1 2023, not to Q4 2023.")
        print()
    
    for i, (quarter, target_arr, yoy_growth) in enumerate(zip(
        ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
        yoy_targets,
        yoy_predictions
    )):
        print(f"{quarter}: ${target_arr:,.0f} ({yoy_growth*100:.1f}% YoY growth)")
    
    # Summary
    starting_arr = last_arr
    ending_arr = yoy_targets[-1]
    total_growth = ((ending_arr - starting_arr) / starting_arr) * 100
    
    print(f"\nSUMMARY:")
    print(f"  Starting ARR (Q4 2023): ${starting_arr:,.0f}")
    print(f"  Ending ARR (Q4 2024): ${ending_arr:,.0f}")
    print(f"  Total Growth: {total_growth:.1f}%")
    print(f"  Growth Path: Pure model predictions, no constraints")

if __name__ == "__main__":
    test_new_company()

