#!/usr/bin/env python3
"""
Generate multiple test cases with different company profiles to validate our model.
"""

import pandas as pd
import numpy as np
import pickle
from financial_forecasting_model import load_and_clean_data, engineer_features
from enhanced_guided_input import EnhancedGuidedInputSystem

def load_single_quarter_models():
    """Load the single-quarter models."""
    try:
        with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return None

def create_test_company_data(name, arr_q4, net_new_arr_q4, headcount, gross_margin, quarters=4):
    """Create test company data for a specific scenario."""
    # Calculate growth rate
    growth_rate = (net_new_arr_q4 / (arr_q4 - net_new_arr_q4)) * 100
    
    # Create quarterly progression
    quarters_data = []
    current_arr = arr_q4 - net_new_arr_q4  # Start from Q3 ARR
    
    for i in range(quarters):
        quarter_name = f"Q{i+1} 2023"
        current_arr += net_new_arr_q4 if i == quarters-1 else net_new_arr_q4 * (0.8 + i * 0.1)  # Gradual growth
        
        quarters_data.append({
            'Quarter': quarter_name,
            'ARR_End_of_Quarter': current_arr,
            'Quarterly_Net_New_ARR': net_new_arr_q4 * (0.8 + i * 0.1) if i < quarters-1 else net_new_arr_q4,
            'QRR_Quarterly_Recurring_Revenue': current_arr * 0.25,  # Assume 25% of ARR
            'Headcount': headcount + i * 5,  # Gradual headcount growth
            'Gross_Margin_Percent': gross_margin,
            'Net_Profit_Loss_Margin_Percent': -50 - i * 5  # Improving margins
        })
    
    return pd.DataFrame(quarters_data)

def predict_company_forecast(model_data, company_data, company_name):
    """Predict forecast for a specific company."""
    print(f"\n{'='*60}")
    print(f"🏢 TESTING: {company_name}")
    print(f"{'='*60}")
    
    # Show company profile
    latest_q = company_data.iloc[-1]
    current_arr = latest_q['ARR_End_of_Quarter']
    net_new_arr = latest_q['Quarterly_Net_New_ARR']
    growth_rate = (net_new_arr / (current_arr - net_new_arr)) * 100
    
    print(f"📊 Company Profile:")
    print(f"  Current ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    print(f"  Growth Rate: {growth_rate:.1f}%")
    print(f"  Headcount: {latest_q['Headcount']}")
    print(f"  Gross Margin: {latest_q['Gross_Margin_Percent']}%")
    
    # Prepare data with smart defaults
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    
    # Rename columns to match expected format
    df_renamed = company_data.copy()
    df_renamed['Financial Quarter'] = df_renamed['Quarter']
    df_renamed['cARR'] = df_renamed['ARR_End_of_Quarter']
    df_renamed['Net New ARR'] = df_renamed['Quarterly_Net_New_ARR']
    df_renamed['Headcount (HC)'] = df_renamed['Headcount']
    df_renamed['Gross Margin (in %)'] = df_renamed['Gross_Margin_Percent']
    df_renamed['id_company'] = company_name
    df_renamed['Revenue'] = df_renamed['cARR']
    df_renamed['ARR YoY Growth (in %)'] = 0
    df_renamed['Revenue YoY Growth (in %)'] = 0
    df_renamed['Cash Burn (OCF & ICF)'] = -df_renamed['cARR'] * 0.3
    df_renamed['Sales & Marketing'] = df_renamed['cARR'] * 0.2
    df_renamed['Expansion & Upsell'] = df_renamed['cARR'] * 0.1
    df_renamed['Churn & Reduction'] = -df_renamed['cARR'] * 0.05
    df_renamed['Customers (EoP)'] = df_renamed['Headcount (HC)'] * 10
    
    # Process the data
    df_clean = df_renamed.copy()
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})|(\d{4})')[0].fillna(df_clean['Financial Quarter'].str.extract(r'(\d{4})')[0]).astype(int)
    df_clean['Quarter Num'] = df_clean['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_clean['time_idx'] = df_clean['Year'] * 4 + df_clean['Quarter Num']
    df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Feature engineering
    processed_df = engineer_features(df_clean)
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # Prepare features
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    X_predict = pd.DataFrame(index=prediction_input_row.index, columns=feature_cols)
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            X_predict[col] = 0
    
    X_predict = X_predict.fillna(0)
    
    # Make predictions
    current_arr = prediction_input_row['cARR'].iloc[0]
    future_quarters = ["FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4"]
    forecast = []
    
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        model = models[quarter]
        predicted_yoy_growth = model.predict(X_predict)[0]
        quarterly_growth_rate = ((1 + predicted_yoy_growth/100) ** (1/4) - 1) * 100
        predicted_arr = current_arr * (1 + quarterly_growth_rate/100)
        
        uncertainty_factor = 0.10
        lower_bound = predicted_arr * (1 - uncertainty_factor)
        upper_bound = predicted_arr * (1 + uncertainty_factor)
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound,
            "Quarterly Growth (%)": quarterly_growth_rate,
            "YoY Growth (%)": predicted_yoy_growth
        })
        
        current_arr = predicted_arr
    
    result_df = pd.DataFrame(forecast)
    
    # Show results
    print(f"\n🎯 PREDICTION RESULTS:")
    print(result_df[['Future Quarter', 'Quarterly Growth (%)', 'Predicted ARR ($)']])
    
    # Summary
    final_arr = result_df.iloc[-1]['Predicted ARR ($)']
    total_growth = ((final_arr - current_arr) / current_arr) * 100
    print(f"\n📊 SUMMARY:")
    print(f"  Total Growth: {total_growth:.1f}% over 4 quarters")
    print(f"  Final ARR: ${final_arr:,.0f}")
    
    return result_df

def main():
    """Generate and test multiple company scenarios."""
    print("🧪 GENERATING MULTIPLE TEST CASES")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    print("📊 MODEL PERFORMANCE:")
    for quarter, r2 in model_data['r2_scores'].items():
        print(f"{quarter}: R² = {r2:.4f}")
    print(f"Overall R²: {np.mean(list(model_data['r2_scores'].values())):.4f}")
    
    # Define test cases
    test_cases = [
        {
            "name": "High-Growth Startup",
            "arr_q4": 5000000,
            "net_new_arr_q4": 1500000,  # 30% growth
            "headcount": 50,
            "gross_margin": 75
        },
        {
            "name": "Moderate Growth SaaS",
            "arr_q4": 10000000,
            "net_new_arr_q4": 2000000,  # 20% growth
            "headcount": 100,
            "gross_margin": 80
        },
        {
            "name": "Mature Enterprise",
            "arr_q4": 50000000,
            "net_new_arr_q4": 5000000,  # 10% growth
            "headcount": 500,
            "gross_margin": 85
        },
        {
            "name": "Hyper-Growth Unicorn",
            "arr_q4": 20000000,
            "net_new_arr_q4": 8000000,  # 40% growth
            "headcount": 200,
            "gross_margin": 70
        },
        {
            "name": "Early Stage Startup",
            "arr_q4": 1000000,
            "net_new_arr_q4": 400000,  # 40% growth
            "headcount": 20,
            "gross_margin": 65
        },
        {
            "name": "Stable Growth Company",
            "arr_q4": 25000000,
            "net_new_arr_q4": 3000000,  # 12% growth
            "headcount": 300,
            "gross_margin": 82
        }
    ]
    
    # Test each case
    results = {}
    for case in test_cases:
        # Create test data
        company_data = create_test_company_data(
            case["name"],
            case["arr_q4"],
            case["net_new_arr_q4"],
            case["headcount"],
            case["gross_margin"]
        )
        
        # Make prediction
        forecast = predict_company_forecast(model_data, company_data, case["name"])
        results[case["name"]] = forecast
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON OF ALL TEST CASES")
    print(f"{'='*80}")
    print(f"{'Company':<25} {'Current ARR':<15} {'Growth Rate':<12} {'Total Growth':<12} {'Final ARR':<15}")
    print("-" * 80)
    
    for case in test_cases:
        name = case["name"]
        current_arr = case["arr_q4"]
        growth_rate = (case["net_new_arr_q4"] / (case["arr_q4"] - case["net_new_arr_q4"])) * 100
        final_arr = results[name].iloc[-1]['Predicted ARR ($)']
        total_growth = ((final_arr - current_arr) / current_arr) * 100
        
        print(f"{name:<25} ${current_arr:>12,.0f} {growth_rate:>10.1f}% {total_growth:>10.1f}% ${final_arr:>12,.0f}")
    
    print(f"\n✅ All test cases completed!")

if __name__ == "__main__":
    main()


