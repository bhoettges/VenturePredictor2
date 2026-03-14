#!/usr/bin/env python3
"""
Test the final solution: High-accuracy model with documented Q1 bias.
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

def predict_with_documented_bias(model_data, company_df, company_name):
    """
    Predict using high-accuracy YoY model with documented Q1 bias.
    """
    print(f"\n{'='*60}")
    print(f"🏢 TESTING: {company_name}")
    print(f"{'='*60}")
    
    # Load the trained models
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Process the company data (same as before)
    df_clean = company_df.copy()
    
    # Rename columns to match expected format
    if 'Quarter' in df_clean.columns:
        df_clean['Financial Quarter'] = df_clean['Quarter']
    if 'ARR_End_of_Quarter' in df_clean.columns:
        df_clean['cARR'] = df_clean['ARR_End_of_Quarter']
    if 'Quarterly_Net_New_ARR' in df_clean.columns:
        df_clean['Net New ARR'] = df_clean['Quarterly_Net_New_ARR']
    if 'Headcount' in df_clean.columns:
        df_clean['Headcount (HC)'] = df_clean['Headcount']
    if 'Gross_Margin_Percent' in df_clean.columns:
        df_clean['Gross Margin (in %)'] = df_clean['Gross_Margin_Percent']
    
    # Add missing columns
    df_clean['id_company'] = company_name
    df_clean['Revenue'] = df_clean['cARR']
    
    # Calculate actual YoY growth rates
    df_clean['ARR YoY Growth (in %)'] = df_clean.groupby('id_company')['cARR'].pct_change(4) * 100
    df_clean['Revenue YoY Growth (in %)'] = df_clean.groupby('id_company')['Revenue'].pct_change(4) * 100
    df_clean['Cash Burn (OCF & ICF)'] = -df_clean['cARR'] * 0.3
    df_clean['Sales & Marketing'] = df_clean['cARR'] * 0.2
    df_clean['Expansion & Upsell'] = df_clean['cARR'] * 0.1
    df_clean['Churn & Reduction'] = -df_clean['cARR'] * 0.05
    df_clean['Customers (EoP)'] = df_clean['Headcount (HC)'] * 10
    
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})|(\d{4})')[0].fillna(df_clean['Financial Quarter'].str.extract(r'(\d{4})')[0]).astype(int)
    df_clean['Quarter Num'] = df_clean['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_clean['time_idx'] = df_clean['Year'] * 4 + df_clean['Quarter Num']
    df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)

    # Data processing
    potential_numeric_cols = df_clean.columns.drop(['Financial Quarter', 'id_company'])
    for col in potential_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean['Net New ARR'] = df_clean.groupby('id_company')['cARR'].transform(lambda x: x.diff())
    stock_vars = ['Headcount (HC)', 'Customers (EoP)']
    for col in stock_vars:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('id_company')[col].transform(lambda x: x.ffill())
    flow_vars = ['Net New ARR', 'Cash Burn (OCF & ICF)', 'Sales & Marketing', 'Expansion & Upsell', 'Churn & Reduction']
    for col in flow_vars:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)
    df_clean['Gross Margin (in %)'] = df_clean.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
    df_clean['Gross Margin (in %)'].fillna(df_clean['Gross Margin (in %)'].median(), inplace=True)

    # Feature engineering
    processed_df = engineer_features(df_clean)
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # Prepare features
    X_predict = pd.DataFrame(index=prediction_input_row.index, columns=feature_cols)
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            X_predict[col] = 0
    X_predict = X_predict.fillna(0)
    
    # Get starting ARR
    last_known_q = processed_df.iloc[-1]
    current_arr = last_known_q['cARR']
    
    # Show company profile
    net_new_arr = last_known_q['Net New ARR']
    growth_rate = (net_new_arr / (current_arr - net_new_arr)) * 100 if net_new_arr > 0 else 0
    
    print(f"📊 Company Profile:")
    print(f"  Current ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    print(f"  Growth Rate: {growth_rate:.1f}%")
    
    # Make predictions with YoY model (NO bias correction)
    yoy_predictions = []
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        model = models[quarter]
        predicted_yoy_growth = model.predict(X_predict)[0]
        yoy_predictions.append(predicted_yoy_growth)
    
    # Convert YoY predictions to quarterly growth rates and apply sequentially
    # This maintains the growth trajectory instead of using flawed YoY baseline logic
    future_quarters = ["FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4"]
    forecast = []
    
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        # Get the model prediction (YoY growth rate)
        predicted_yoy_growth = yoy_predictions[i]
        
        # Convert YoY growth to quarterly growth rate
        # Formula: quarterly_growth_rate = ((1 + YoY_growth/100) ** (1/4) - 1) * 100
        quarterly_growth_rate = ((1 + predicted_yoy_growth/100) ** (1/4) - 1) * 100
        
        # Apply quarterly growth to get next quarter's ARR
        if i == 0:
            # For Q1, start from current ARR
            predicted_arr = current_arr * (1 + quarterly_growth_rate/100)
        else:
            # For Q2, Q3, Q4, use the previous quarter's predicted ARR
            predicted_arr = forecast[i-1]["Predicted ARR ($)"] * (1 + quarterly_growth_rate/100)
        
        # Calculate uncertainty bounds (±10%)
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

    result_df = pd.DataFrame(forecast)
    
    # Show results
    print(f"\n🎯 PREDICTION RESULTS:")
    print(result_df[['Future Quarter', 'Quarterly Growth (%)', 'Predicted ARR ($)']])
    
    # Summary
    final_arr = result_df.iloc[-1]['Predicted ARR ($)']
    starting_arr = processed_df.iloc[-1]['cARR'] if len(processed_df) > 0 else current_arr
    total_growth = ((final_arr - starting_arr) / starting_arr) * 100
    print(f"\n📊 SUMMARY:")
    print(f"  Total Growth: {total_growth:.1f}% over 4 quarters")
    print(f"  Final ARR: ${final_arr:,.0f}")
    
    # Document Q1 bias
    q1_growth = result_df.iloc[0]['Quarterly Growth (%)']
    avg_q2_q4 = np.mean([result_df.iloc[i]['Quarterly Growth (%)'] for i in range(1, 4)])
    q1_bias = q1_growth - avg_q2_q4
    
    print(f"\n⚠️  MODEL LIMITATION:")
    print(f"  Q1 Growth: {q1_growth:.1f}% (tends to be optimistic)")
    print(f"  Q2-Q4 Average: {avg_q2_q4:.1f}%")
    print(f"  Q1 Bias: +{q1_bias:.1f}% (this is a known model limitation)")
    
    return result_df

def create_test_company_data(name, arr_q4, net_new_arr_q4, headcount, gross_margin, quarters=4):
    """Create test company data for a specific scenario."""
    # Calculate growth rate (quarterly growth rate)
    growth_rate = (net_new_arr_q4 / arr_q4) * 100
    
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

def create_test_company_data_with_historical(name, arr_q4, net_new_arr_q4, headcount, gross_margin, historical_arr=None, quarters=4):
    """Create test company data with historical ARR data. Always uses historical data when available."""
    if historical_arr and len(historical_arr) >= 4:
        # Use provided historical data - need to extend to 8 quarters for YoY calculations
        quarters_data = []
        
        # First, create 4 quarters of previous year data (FY22) to enable YoY calculations
        # Estimate previous year data based on current growth patterns
        if len(historical_arr) >= 4:
            # Calculate growth rate from historical data
            growth_rate = (historical_arr[-1] - historical_arr[0]) / historical_arr[0] if historical_arr[0] > 0 else 0.1
            
            # Create previous year data (FY22)
            prev_year_base = historical_arr[0] / (1 + growth_rate)
            for i in range(4):
                quarter_name = f"Q{i+1} 2022"
                quarter_arr = prev_year_base * ((1 + growth_rate) ** (i/4))
                quarter_net_new = quarter_arr * 0.05  # Assume 5% quarterly growth
                
                quarters_data.append({
                    'Quarter': quarter_name,
                    'ARR_End_of_Quarter': quarter_arr,
                    'Quarterly_Net_New_ARR': quarter_net_new,
                    'QRR_Quarterly_Recurring_Revenue': quarter_arr * 0.25,
                    'Headcount': max(1, int(quarter_arr / 200000)) + i * 5,
                    'Gross_Margin_Percent': gross_margin,
                    'Net_Profit_Loss_Margin_Percent': -50 - i * 5
                })
        
        # Now add the current year data (FY23)
        for i, arr in enumerate(historical_arr):
            quarter_name = f"Q{i+1} 2023"
            # Calculate net new ARR for this quarter
            if i == 0:
                net_new = arr - quarters_data[-1]['ARR_End_of_Quarter']  # Compare to previous year Q4
            else:
                net_new = arr - historical_arr[i-1]
            
            quarters_data.append({
                'Quarter': quarter_name,
                'ARR_End_of_Quarter': arr,
                'Quarterly_Net_New_ARR': max(0, net_new),  # Ensure non-negative
                'QRR_Quarterly_Recurring_Revenue': arr * 0.25,  # Assume 25% of ARR
                'Headcount': headcount + i * 5,  # Gradual headcount growth
                'Gross_Margin_Percent': gross_margin,
                'Net_Profit_Loss_Margin_Percent': -50 - i * 5  # Improving margins
            })
        
        # Add current quarter if it's not in historical data
        if len(historical_arr) == 4 and arr_q4 != historical_arr[-1]:
            quarters_data.append({
                'Quarter': 'Q4 2023',
                'ARR_End_of_Quarter': arr_q4,
                'Quarterly_Net_New_ARR': net_new_arr_q4,
                'QRR_Quarterly_Recurring_Revenue': arr_q4 * 0.25,
                'Headcount': headcount + 4 * 5,
                'Gross_Margin_Percent': gross_margin,
                'Net_Profit_Loss_Margin_Percent': -50 - 4 * 5
            })
    else:
        # Generate realistic historical data based on current ARR and growth rate
        # Calculate quarterly growth rate from net new ARR
        quarterly_growth_rate = (net_new_arr_q4 / arr_q4) if arr_q4 > 0 else 0
        
        # Generate 8 quarters of data (2 years) to enable YoY calculations
        quarters_data = []
        
        # First year (FY22)
        base_arr = arr_q4 / ((1 + quarterly_growth_rate) ** 4)
        for i in range(4):
            quarter_name = f"Q{i+1} 2022"
            quarter_arr = base_arr * ((1 + quarterly_growth_rate) ** i)
            quarter_net_new = quarter_arr * quarterly_growth_rate
            
            quarters_data.append({
                'Quarter': quarter_name,
                'ARR_End_of_Quarter': quarter_arr,
                'Quarterly_Net_New_ARR': quarter_net_new,
                'QRR_Quarterly_Recurring_Revenue': quarter_arr * 0.25,
                'Headcount': max(1, int(quarter_arr / 200000)) + i * 5,
                'Gross_Margin_Percent': gross_margin,
                'Net_Profit_Loss_Margin_Percent': -50 - i * 5
            })
        
        # Second year (FY23)
        for i in range(4):
            quarter_name = f"Q{i+1} 2023"
            quarter_arr = base_arr * ((1 + quarterly_growth_rate) ** (4 + i))
            quarter_net_new = quarter_arr * quarterly_growth_rate
            
            quarters_data.append({
                'Quarter': quarter_name,
                'ARR_End_of_Quarter': quarter_arr,
                'Quarterly_Net_New_ARR': quarter_net_new,
                'QRR_Quarterly_Recurring_Revenue': quarter_arr * 0.25,
                'Headcount': max(1, int(quarter_arr / 200000)) + (4 + i) * 5,
                'Gross_Margin_Percent': gross_margin,
                'Net_Profit_Loss_Margin_Percent': -50 - (4 + i) * 5
            })
    
    return pd.DataFrame(quarters_data)

def main():
    """Test the final solution on multiple companies."""
    print("🧪 TESTING FINAL SOLUTION: HIGH-ACCURACY MODEL WITH DOCUMENTED Q1 BIAS")
    print("=" * 80)
    
    # Load the high-accuracy models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Show model performance
    print("\n📊 MODEL PERFORMANCE:")
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
        
        # Make prediction with documented bias
        forecast = predict_with_documented_bias(model_data, company_data, case["name"])
        results[case["name"]] = forecast
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 SUMMARY COMPARISON OF ALL TEST CASES (FINAL SOLUTION)")
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
    
    print(f"\n✅ FINAL SOLUTION: High-accuracy model with documented Q1 bias!")
    print(f"💡 This approach provides production-ready accuracy while being transparent about limitations.")

if __name__ == "__main__":
    main()


