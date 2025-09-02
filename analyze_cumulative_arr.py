#!/usr/bin/env python3
"""
Analyze Cumulative ARR Pattern
=============================

Analyze the cumulative ARR growth pattern for the new company data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_cumulative_arr():
    """Analyze cumulative ARR patterns."""
    print("CUMULATIVE ARR ANALYSIS")
    print("=" * 50)
    
    # Load the company data
    company_df = pd.read_csv('test_company_new.csv')
    
    # Map columns
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    
    print("Company Data:")
    for i, row in company_df.iterrows():
        print(f"  {row['Quarter']}: ARR=${row['cARR']:,.0f}")
    
    # Calculate cumulative ARR (sum of all quarters)
    company_df['Cumulative_ARR'] = company_df['cARR'].cumsum()
    
    print(f"\nCumulative ARR Analysis:")
    for i, row in company_df.iterrows():
        print(f"  {row['Quarter']}: Cumulative ARR=${row['Cumulative_ARR']:,.0f}")
    
    # Calculate cumulative growth rates
    company_df['Cumulative_Growth'] = company_df['Cumulative_ARR'].pct_change(1)
    
    print(f"\nCumulative Growth Rates:")
    for i, row in company_df.iterrows():
        if i > 0:
            growth = row['Cumulative_Growth'] * 100
            print(f"  {row['Quarter']}: {growth:.1f}% cumulative growth")
    
    # Calculate total cumulative ARR for 2023
    total_2023_arr = company_df['Cumulative_ARR'].iloc[-1]
    print(f"\nTotal 2023 Cumulative ARR: ${total_2023_arr:,.0f}")
    
    # Calculate average quarterly contribution
    avg_quarterly_contribution = total_2023_arr / 4
    print(f"Average Quarterly Contribution: ${avg_quarterly_contribution:,.0f}")
    
    # Calculate cumulative growth rate for the year
    q1_arr = company_df['cARR'].iloc[0]
    q4_cumulative = company_df['Cumulative_ARR'].iloc[-1]
    annual_cumulative_growth = (q4_cumulative - q1_arr) / q1_arr * 100
    
    print(f"\nAnnual Cumulative Growth Analysis:")
    print(f"  Q1 ARR: ${q1_arr:,.0f}")
    print(f"  Q4 Cumulative ARR: ${q4_cumulative:,.0f}")
    print(f"  Annual Cumulative Growth: {annual_cumulative_growth:.1f}%")
    
    # Show the pattern
    print(f"\nCumulative ARR Pattern:")
    print(f"  Q1: ${company_df['Cumulative_ARR'].iloc[0]:,.0f} (base)")
    print(f"  Q2: ${company_df['Cumulative_ARR'].iloc[1]:,.0f} (+${company_df['cARR'].iloc[1]:,.0f})")
    print(f"  Q3: ${company_df['Cumulative_ARR'].iloc[2]:,.0f} (+${company_df['cARR'].iloc[2]:,.0f})")
    print(f"  Q4: ${company_df['Cumulative_ARR'].iloc[3]:,.0f} (+${company_df['cARR'].iloc[3]:,.0f})")
    
    return company_df

def predict_cumulative_arr_2024():
    """Predict cumulative ARR for 2024 based on patterns."""
    print(f"\n" + "=" * 50)
    print("CUMULATIVE ARR PREDICTION FOR 2024")
    print("=" * 50)
    
    # Load the company data
    company_df = pd.read_csv('test_company_new.csv')
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    
    # Calculate 2023 cumulative ARR
    total_2023_arr = company_df['cARR'].sum()
    print(f"Total 2023 Cumulative ARR: ${total_2023_arr:,.0f}")
    
    # Different prediction approaches
    print(f"\nPrediction Approaches:")
    
    # 1. Linear growth (same as 2023)
    linear_2024 = total_2023_arr * 2  # Double the cumulative ARR
    print(f"1. Linear Growth (2x 2023): ${linear_2024:,.0f}")
    
    # 2. Conservative growth (1.5x)
    conservative_2024 = total_2023_arr * 1.5
    print(f"2. Conservative Growth (1.5x): ${conservative_2024:,.0f}")
    
    # 3. Aggressive growth (2.5x)
    aggressive_2024 = total_2023_arr * 2.5
    print(f"3. Aggressive Growth (2.5x): ${aggressive_2024:,.0f}")
    
    # 4. Based on Q4 momentum
    q4_2023_arr = company_df['cARR'].iloc[-1]
    q4_momentum_2024 = total_2023_arr + (q4_2023_arr * 4)  # Q4 run rate for full year
    print(f"4. Q4 Momentum (Q4 run rate): ${q4_momentum_2024:,.0f}")
    
    # 5. Average quarterly contribution approach
    avg_quarterly = total_2023_arr / 4
    avg_quarterly_2024 = avg_quarterly * 8  # 8 quarters total
    print(f"5. Average Quarterly (8 quarters): ${avg_quarterly_2024:,.0f}")
    
    return {
        'linear': linear_2024,
        'conservative': conservative_2024,
        'aggressive': aggressive_2024,
        'q4_momentum': q4_momentum_2024,
        'avg_quarterly': avg_quarterly_2024
    }

if __name__ == "__main__":
    # Analyze current data
    company_df = analyze_cumulative_arr()
    
    # Predict 2024
    predictions = predict_cumulative_arr_2024()
    
    print(f"\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    print("Consider building a model that predicts:")
    print("1. Total cumulative ARR for 2024")
    print("2. Quarterly breakdown of that cumulative ARR")
    print("3. This would be more stable than QoQ/YoY predictions")
    print("4. Better for business planning and resource allocation")

