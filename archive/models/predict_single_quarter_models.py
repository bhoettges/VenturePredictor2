#!/usr/bin/env python3
"""
Predict using single-quarter models for high accuracy.
Each model predicts only 1 quarter ahead, then we chain the predictions.
"""

import pandas as pd
import numpy as np
import pickle
from financial_forecasting_model import load_and_clean_data, engineer_features

def load_single_quarter_models():
    """Load the single-quarter models."""
    try:
        with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Single-quarter models loaded successfully")
        return model_data
    except FileNotFoundError:
        print("❌ ERROR: Model file not found. Please run retrain_single_quarter_models.py first.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return None

def predict_with_single_quarter_models(model_data, company_df):
    """
    Predicts ARR progression using single-quarter models.
    Each model predicts 1 quarter ahead, then we chain the predictions.
    """
    print("Step 1: Loading and preprocessing company data...")
    
    # Load the trained models
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Process the company data directly
    print("Step 1.1: Processing company data...")
    
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
    print("Step 2: Engineering features...")
    processed_df = engineer_features(df_clean)
    
    # Get the last quarter's data for prediction
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # --- Feature Imputation (using smart defaults from enhanced_guided_input) ---
    print("Step 1.5: Using smart defaults from enhanced_guided_input...")
    
    # The enhanced_guided_input system already provides smart defaults
    # We just need to ensure all required features are present
    X_predict = pd.DataFrame(index=prediction_input_row.index, columns=feature_cols)
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            # Use 0 for any missing features (enhanced_guided_input should have provided all needed features)
            X_predict[col] = 0
            print(f"⚠️  Warning: Missing feature '{col}', using 0")

    # Fill any remaining NaNs with 0
    X_predict = X_predict.fillna(0)
    
    print(f"🔍 Debug: Input features shape: {X_predict.shape}")
    print(f"🔍 Debug: Sample input values: {X_predict.iloc[0, :5].to_dict()}")

    print("Step 2: Making predictions with single-quarter models...")
    
    # Get starting ARR
    last_known_q = processed_df.iloc[-1]
    current_arr = last_known_q['cARR']
    print(f"🔍 Starting ARR: ${current_arr:,.0f}")
    
    # Predict each quarter sequentially
    future_quarters = ["FY26 Q1", "FY26 Q2", "FY26 Q3", "FY26 Q4"]
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
    """Test the single-quarter model prediction with sample data."""
    print("🧪 TESTING SINGLE-QUARTER MODEL PREDICTION")
    print("=" * 50)
    
    # Load the models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Show model performance
    print("\n📊 MODEL PERFORMANCE:")
    for quarter, r2 in model_data['r2_scores'].items():
        print(f"{quarter}: R² = {r2:.4f}")
    print(f"Overall R²: {np.mean(list(model_data['r2_scores'].values())):.4f}")
    
    # Create sample company data
    from enhanced_guided_input import EnhancedGuidedInputSystem
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    input_data = system.create_forecast_input_with_history(2800000, 800000)  # 40% quarterly growth
    
    # Make prediction
    result = predict_with_single_quarter_models(model_data, input_data)
    
    print("\n🎯 PREDICTION RESULTS (SINGLE-QUARTER MODELS):")
    print(result)
    
    print("\n📊 SUMMARY:")
    current_arr = 2800000
    final_arr = result.iloc[-1]['Predicted ARR ($)']
    total_growth = ((final_arr - current_arr) / current_arr) * 100
    print(f"Current ARR: ${current_arr:,.0f}")
    print(f"Final ARR (Q4): ${final_arr:,.0f}")
    print(f"Total Growth: {total_growth:.1f}% over 4 quarters")

if __name__ == "__main__":
    main()
