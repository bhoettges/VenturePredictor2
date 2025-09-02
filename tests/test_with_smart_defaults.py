#!/usr/bin/env python3
"""
Test the single-quarter models with smart defaults from enhanced_guided_input.
This should fix the conservative predictions.
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
        print("✅ Single-quarter models loaded successfully")
        return model_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return None

def prepare_test_company_with_smart_defaults():
    """Prepare test company data using smart defaults from enhanced_guided_input."""
    print("Step 1: Preparing test company data with smart defaults...")
    
    # Read the CSV file
    df = pd.read_csv('test_company_2024.csv')
    print(f"📊 Loaded {len(df)} quarters of data")
    
    # Get the latest quarter data
    latest_q = df.iloc[-1]
    current_arr = latest_q['ARR_End_of_Quarter']
    net_new_arr = latest_q['Quarterly_Net_New_ARR']
    
    print(f"📊 Latest Quarter Data:")
    print(f"  ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    print(f"  Growth Rate: {(net_new_arr/(current_arr-net_new_arr))*100:.1f}%")
    
    # Initialize the enhanced guided input system
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    
    # Create forecast input with smart defaults
    print("Step 2: Generating smart defaults...")
    input_data = system.create_forecast_input_with_history(current_arr, net_new_arr)
    
    print(f"✅ Generated {len(input_data)} quarters of data with smart defaults")
    return input_data

def predict_with_smart_defaults(model_data, company_df):
    """
    Predicts ARR progression using single-quarter models with smart defaults.
    """
    print("Step 3: Processing company data for prediction...")
    
    # Load the trained models
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Apply the same cleaning logic as in load_and_clean_data
    df_clean = company_df.copy()
    
    # --- Time Index Creation (must match training exactly) ---
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df_clean['Year'] = df_clean['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df_clean['Quarter Num'] = df_clean['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_clean['time_idx'] = df_clean['Year'] * 4 + df_clean['Quarter Num']
    df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)

    # --- Data Type Coercion ---
    potential_numeric_cols = df_clean.columns.drop(['Financial Quarter', 'id_company'])
    for col in potential_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # --- Nuanced Imputation Strategy (must match training exactly) ---
    df_clean['Net New ARR'] = df_clean.groupby('id_company')['cARR'].transform(lambda x: x.diff())

    # Forward-fill stock variables
    stock_vars = ['Headcount (HC)', 'Customers (EoP)']
    for col in stock_vars:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('id_company')[col].transform(lambda x: x.ffill())

    # Fill flow variables with 0
    flow_vars = ['Net New ARR', 'Cash Burn (OCF & ICF)', 'Sales & Marketing', 'Expansion & Upsell', 'Churn & Reduction']
    for col in flow_vars:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)

    # Impute metrics with company-specific median
    df_clean['Gross Margin (in %)'] = df_clean.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
    df_clean['Gross Margin (in %)'].fillna(df_clean['Gross Margin (in %)'].median(), inplace=True)

    # --- Feature Engineering ---
    print("Step 4: Engineering features...")
    processed_df = engineer_features(df_clean)
    
    # Get the last quarter's data for prediction
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # --- Feature Imputation with Smart Defaults ---
    print("Step 5: Preparing features with smart defaults...")
    X_predict = pd.DataFrame(index=prediction_input_row.index, columns=feature_cols)
    
    missing_count = 0
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            # Use 0 for any missing features (this is the problem!)
            X_predict[col] = 0
            missing_count += 1

    # Fill any remaining NaNs with 0
    X_predict = X_predict.fillna(0)
    
    print(f"🔍 Debug: Input features shape: {X_predict.shape}")
    print(f"🔍 Debug: Missing features: {missing_count}/{len(feature_cols)} ({missing_count/len(feature_cols)*100:.1f}%)")
    
    # Show some key feature values
    key_features = ['cARR', 'Net New ARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Sales & Marketing', 'Cash Burn (OCF & ICF)']
    print(f"🔍 Debug: Key feature values:")
    for feature in key_features:
        if feature in X_predict.columns:
            value = X_predict[feature].iloc[0]
            print(f"  {feature}: {value}")

    print("Step 6: Making predictions with single-quarter models...")
    
    # Get starting ARR
    last_known_q = processed_df.iloc[-1]
    current_arr = last_known_q['cARR']
    print(f"🔍 Starting ARR: ${current_arr:,.0f}")
    
    # Calculate actual quarterly growth rate from the data
    if len(processed_df) >= 2:
        prev_arr = processed_df.iloc[-2]['cARR']
        actual_qoq_growth = ((current_arr - prev_arr) / prev_arr) * 100
        print(f"🔍 Actual Q4 2023 QoQ Growth: {actual_qoq_growth:.1f}%")
    
    # Predict each quarter sequentially
    future_quarters = ["FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4"]
    forecast = []
    
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        # Get the model for this quarter
        model = models[quarter]
        
        # Make prediction (this is YoY growth rate)
        predicted_yoy_growth = model.predict(X_predict)[0]
        print(f"🔍 {quarter} Model Prediction: {predicted_yoy_growth:.1f}% YoY growth")
        
        # CRITICAL FIX: Convert YoY growth to quarterly growth rate
        # YoY growth needs to be converted to quarterly growth rate
        # Formula: quarterly_growth_rate = ((1 + YoY_growth/100) ** (1/4) - 1) * 100
        quarterly_growth_rate = ((1 + predicted_yoy_growth/100) ** (1/4) - 1) * 100
        
        # Apply quarterly growth to get next quarter's ARR
        predicted_arr = current_arr * (1 + quarterly_growth_rate/100)
        
        # Calculate uncertainty bounds (±10%)
        uncertainty_factor = 0.10
        lower_bound = predicted_arr * (1 - uncertainty_factor)
        upper_bound = predicted_arr * (1 + uncertainty_factor)
        
        print(f"🔍 {quarter} Debug: YoY Growth = {predicted_yoy_growth:.1f}% → QoQ Growth = {quarterly_growth_rate:.1f}% → ARR = ${predicted_arr:,.0f}")
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound,
            "Quarterly Growth (%)": quarterly_growth_rate,
            "YoY Growth (%)": predicted_yoy_growth
        })
        
        # Update current ARR for next iteration
        current_arr = predicted_arr

    return pd.DataFrame(forecast)

def main():
    """Test the single-quarter models with smart defaults."""
    print("🧪 TESTING SINGLE-QUARTER MODELS WITH SMART DEFAULTS")
    print("=" * 60)
    
    # Load the models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Show model performance
    print("\n📊 MODEL PERFORMANCE:")
    for quarter, r2 in model_data['r2_scores'].items():
        print(f"{quarter}: R² = {r2:.4f}")
    print(f"Overall R²: {np.mean(list(model_data['r2_scores'].values())):.4f}")
    
    # Prepare test company data with smart defaults
    test_data = prepare_test_company_with_smart_defaults()
    
    # Make prediction
    result = predict_with_smart_defaults(model_data, test_data)
    
    print("\n🎯 PREDICTION RESULTS WITH SMART DEFAULTS:")
    print("=" * 50)
    print(result)
    
    print("\n📊 SUMMARY:")
    current_arr = 2800000  # Q4 2023 ARR
    final_arr = result.iloc[-1]['Predicted ARR ($)']
    total_growth = ((final_arr - current_arr) / current_arr) * 100
    print(f"Current ARR (Q4 2023): ${current_arr:,.0f}")
    print(f"Final ARR (Q4 2024): ${final_arr:,.0f}")
    print(f"Total Growth: {total_growth:.1f}% over 4 quarters")
    
    # Show quarterly growth pattern
    print(f"\n📈 QUARTERLY GROWTH PATTERN:")
    for _, row in result.iterrows():
        print(f"{row['Future Quarter']}: {row['Quarterly Growth (%)']:.1f}% growth → ${row['Predicted ARR ($)']:,.0f}")

if __name__ == "__main__":
    main()


