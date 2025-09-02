#!/usr/bin/env python3
"""
Analyze Model Accuracy Issues
============================

Analyze why the cumulative ARR model is significantly underestimating actual growth.
"""

import pandas as pd
import numpy as np

def analyze_actual_vs_predicted():
    """Analyze the actual vs predicted performance."""
    print("MODEL ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Actual company data
    actual_data = {
        'Q4 2022': 1700000,
        'Q1 2023': 657000,
        'Q2 2023': 401000,
        'Q3 2023': 375207,
        'Q4 2023': 785352,
        'Q4 2024': 7800000  # Actual cumulative ARR at end of 2024
    }
    
    # Model predictions
    predicted_data = {
        'Q4 2022': 1700000,
        'Q1 2023': 657000,
        'Q2 2023': 401000,
        'Q3 2023': 375207,
        'Q4 2023': 785352,
        'Q4 2024': 5841477  # Model prediction
    }
    
    print("ACTUAL vs PREDICTED COMPARISON:")
    print("-" * 40)
    print(f"{'Quarter':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
    print("-" * 40)
    
    for quarter in actual_data:
        actual = actual_data[quarter]
        predicted = predicted_data[quarter]
        error = actual - predicted
        error_pct = (error / actual) * 100 if actual != 0 else 0
        
        print(f"{quarter:<12} ${actual:<11,.0f} ${predicted:<11,.0f} ${error:<11,.0f} {error_pct:<9.1f}%")
    
    # Calculate growth rates
    print(f"\nGROWTH RATE ANALYSIS:")
    print("-" * 40)
    
    # 2023 growth (Q4 2022 to Q4 2023)
    actual_2023_growth = (actual_data['Q4 2023'] - actual_data['Q4 2022']) / actual_data['Q4 2022'] * 100
    predicted_2023_growth = (predicted_data['Q4 2023'] - predicted_data['Q4 2022']) / predicted_data['Q4 2022'] * 100
    
    print(f"2023 Growth (Q4 2022 → Q4 2023):")
    print(f"  Actual: {actual_2023_growth:.1f}%")
    print(f"  Predicted: {predicted_2023_growth:.1f}%")
    print(f"  Difference: {actual_2023_growth - predicted_2023_growth:.1f}%")
    
    # 2024 growth (Q4 2023 to Q4 2024)
    actual_2024_growth = (actual_data['Q4 2024'] - actual_data['Q4 2023']) / actual_data['Q4 2023'] * 100
    predicted_2024_growth = (predicted_data['Q4 2024'] - predicted_data['Q4 2023']) / predicted_data['Q4 2023'] * 100
    
    print(f"\n2024 Growth (Q4 2023 → Q4 2024):")
    print(f"  Actual: {actual_2024_growth:.1f}%")
    print(f"  Predicted: {predicted_2024_growth:.1f}%")
    print(f"  Difference: {actual_2024_growth - predicted_2024_growth:.1f}%")
    
    # Total 2-year growth
    actual_total_growth = (actual_data['Q4 2024'] - actual_data['Q4 2022']) / actual_data['Q4 2022'] * 100
    predicted_total_growth = (predicted_data['Q4 2024'] - predicted_data['Q4 2022']) / predicted_data['Q4 2022'] * 100
    
    print(f"\nTotal 2-Year Growth (Q4 2022 → Q4 2024):")
    print(f"  Actual: {actual_total_growth:.1f}%")
    print(f"  Predicted: {predicted_total_growth:.1f}%")
    print(f"  Difference: {actual_total_growth - predicted_total_growth:.1f}%")
    
    # Calculate what the model should have predicted
    print(f"\n" + "=" * 50)
    print("WHAT THE MODEL SHOULD HAVE PREDICTED")
    print("=" * 50)
    
    # If we know the actual 2024 growth rate, what should the quarterly breakdown be?
    actual_2024_growth_rate = actual_2024_growth / 100  # Convert to decimal
    
    print(f"Actual 2024 growth rate: {actual_2024_growth:.1f}%")
    print(f"This means the model should have predicted:")
    print(f"  Q4 2024 Cumulative ARR: ${actual_data['Q4 2024']:,.0f}")
    print(f"  vs Model's prediction: ${predicted_data['Q4 2024']:,.0f}")
    print(f"  Underestimation: ${actual_data['Q4 2024'] - predicted_data['Q4 2024']:,.0f}")
    
    # Analyze the quarterly pattern
    print(f"\nQUARTERLY PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Calculate actual quarterly contributions for 2024
    # We need to estimate what the quarterly breakdown might have been
    # Let's assume a reasonable quarterly progression
    
    # Estimate quarterly ARR for 2024 based on the total growth
    # This is a rough estimate since we only have the end-of-year number
    
    # If we assume the company grew steadily throughout 2024
    # and ended at $7.8M cumulative, what might the quarterly breakdown look like?
    
    print("Estimated quarterly ARR contributions for 2024:")
    print("(Based on reaching $7.8M cumulative by end of 2024)")
    
    # Let's assume a reasonable quarterly progression
    # Q1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.0M (example)
    estimated_quarterly = {
        'Q1 2024': 1200000,
        'Q2 2024': 1500000, 
        'Q3 2024': 1800000,
        'Q4 2024': 2000000
    }
    
    cumulative_2023 = actual_data['Q4 2023']
    
    print(f"Starting point (Q4 2023): ${cumulative_2023:,.0f}")
    for quarter, arr in estimated_quarterly.items():
        cumulative_2023 += arr
        print(f"{quarter}: ${arr:,.0f} → Cumulative: ${cumulative_2023:,.0f}")
    
    print(f"\nThis would result in Q4 2024 cumulative ARR of: ${cumulative_2023:,.0f}")
    print(f"Actual Q4 2024: ${actual_data['Q4 2024']:,.0f}")
    
    return {
        'actual_data': actual_data,
        'predicted_data': predicted_data,
        'actual_2024_growth': actual_2024_growth,
        'predicted_2024_growth': predicted_2024_growth
    }

def identify_model_issues():
    """Identify potential issues with the model."""
    print(f"\n" + "=" * 50)
    print("POTENTIAL MODEL ISSUES")
    print("=" * 50)
    
    issues = [
        "1. **Training Data Bias**: The model may be trained on companies with lower growth rates",
        "2. **Feature Mismatch**: The feature completion system may not be capturing the right patterns",
        "3. **Target Variable Issues**: Cumulative ARR growth may not be the right target",
        "4. **Data Quality**: The training data may not include enough high-growth companies",
        "5. **Model Complexity**: The model may be too simple to capture complex growth patterns",
        "6. **Temporal Issues**: The model may not be capturing recent growth trends",
        "7. **Outlier Handling**: The model may be capping growth rates too aggressively"
    ]
    
    for issue in issues:
        print(issue)
    
    print(f"\nRECOMMENDATIONS:")
    print("-" * 30)
    print("1. **Retrain with More Recent Data**: Include more 2023-2024 data")
    print("2. **Adjust Target Variables**: Consider predicting absolute ARR instead of growth rates")
    print("3. **Improve Feature Engineering**: Add more growth momentum features")
    print("4. **Remove Growth Rate Caps**: Don't cap extreme growth rates")
    print("5. **Use Ensemble Methods**: Combine multiple models for better predictions")
    print("6. **Add Company-Specific Features**: Include industry, stage, and market factors")

if __name__ == "__main__":
    results = analyze_actual_vs_predicted()
    identify_model_issues()


