#!/usr/bin/env python3
"""
Test script for YallaHabibi Tech company data
"""

import pandas as pd
from financial_prediction import load_trained_model, predict_future_arr

def test_yallahabibi():
    print("🎯 TESTING YALLAHABIBI TECH")
    print("=" * 50)
    
    # Load the company data
    company_df = pd.read_csv('yallahabibi.csv')
    
    # Show latest quarter info
    latest = company_df.iloc[-1]
    print(f"Latest quarter: {latest['Quarter']}")
    print(f"Current ARR: ${latest['ARR_End_of_Quarter']:,}")
    print(f"Net New ARR: ${latest['Quarterly_Net_New_ARR']:,}")
    print(f"QRR: ${latest['QRR_Quarterly_Recurring_Revenue']:,}")
    print(f"Headcount: {latest['Headcount']} employees")
    print(f"Gross Margin: {latest['Gross_Margin_Percent']}%")
    print(f"Net Profit/Loss Margin: {latest['Net_Profit_Loss_Margin_Percent']}%")
    
    print("\n" + "=" * 50)
    
    # Load the trained model
    model_data = load_trained_model()
    
    if model_data is None:
        print("❌ Failed to load model")
        return
    
    # Make predictions
    forecast = predict_future_arr(model_data, company_df)
    
    print("\n🔮 FORECAST RESULTS:")
    print(forecast.to_string(index=False))

if __name__ == "__main__":
    test_yallahabibi()
