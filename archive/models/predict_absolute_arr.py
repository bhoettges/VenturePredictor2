#!/usr/bin/env python3
"""
Predict absolute ARR values for the next 4 quarters using the absolute ARR model.
This is what users actually want to see.
"""

import pandas as pd
import numpy as np
import pickle
from financial_forecasting_model import load_and_clean_data, engineer_features

def load_absolute_arr_model():
    """Load the model trained to predict absolute ARR values."""
    try:
        with open('lightgbm_financial_model_absolute_arr.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model loaded successfully from lightgbm_financial_model_absolute_arr.pkl")
        return model_data
    except FileNotFoundError:
        print("❌ ERROR: Model file not found. Please run retrain_for_absolute_arr.py first.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return None

def predict_absolute_arr(model_data, company_df):
    """
    Predicts absolute ARR values for the next 4 quarters.
    """
    print("Step 1: Loading and preprocessing company data...")
    
    # Load the trained model
    model_pipeline = model_data['model_pipeline']
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
    
    # --- Feature Imputation ---
    print("Step 1.5: Simple feature imputation...")
    
    # Get user's ARR for scaling
    user_arr = prediction_input_row['cARR'].iloc[0] if 'cARR' in prediction_input_row.columns else 1000000
    
    # Simple defaults based on ARR
    simple_defaults = {
        # Core financial metrics
        'cARR': user_arr,
        'ARR YoY Growth (in %)': prediction_input_row['ARR YoY Growth (in %)'].iloc[0] if 'ARR YoY Growth (in %)' in prediction_input_row.columns else 20,
        'Revenue YoY Growth (in %)': 16,  # 80% of ARR growth
        'Gross Margin (in %)': 75,
        'Net Profit/Loss Margin (in %)': -15,
        
        # Headcount and operational metrics (scaled by ARR)
        'Headcount (HC)': max(1, int(user_arr / 150000)),  # Conservative ARR per headcount
        'Customers (EoP)': max(1, int(user_arr / 5000)),  # Conservative customer count
        'Sales & Marketing': user_arr * 0.4,  # 40% of ARR
        'R&D': user_arr * 0.25,  # 25% of ARR
        'G&A': user_arr * 0.15,  # 15% of ARR
        
        # Cash flow metrics
        'Cash Burn (OCF & ICF)': -user_arr * 0.3,  # 30% burn rate
        'Net Cash Flow': -user_arr * 0.25,  # 25% net burn
        
        # Customer metrics
        'Expansion & Upsell': user_arr * 0.1,  # 10% expansion
        'Churn & Reduction': -user_arr * 0.05,  # 5% churn
        
        # Efficiency metrics
        'Magic Number': 0.6,  # Conservative magic number
        'Burn Multiple': 1.5,  # Conservative burn multiple
        'ARR per Headcount': 150000,  # Conservative ARR per headcount
        
        # Time-based features
        'Quarter Num': prediction_input_row['Quarter Num'].iloc[0] if 'Quarter Num' in prediction_input_row.columns else 1,
        'time_idx': prediction_input_row['time_idx'].iloc[0] if 'time_idx' in prediction_input_row.columns else 1,
        
        # Categorical features (use 0 for unknown/neutral)
        'Currency': 0,
        'id_currency': 0,
        'Sector': 0,
        'id_sector': 0,
        'Target Customer': 0,
        'id_target_customer': 0,
        'Country': 0,
        'id_country': 0,
        'Deal Team': 0,
        'id_deal_team': 0
    }
    
    # Create feature matrix
    X_predict = pd.DataFrame(index=[0], columns=feature_cols)
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        elif col in simple_defaults:
            X_predict[col] = simple_defaults[col]
            print(f"📊 Using simple default for '{col}': {simple_defaults[col]}")
        else:
            # For unknown features, use 0 but log it
            X_predict[col] = 0
            print(f"⚠️  Warning: Unknown feature '{col}', using 0")

    # Fill any remaining NaNs with 0
    X_predict = X_predict.fillna(0)
    
    print(f"🔍 Debug: Input features shape: {X_predict.shape}")
    print(f"🔍 Debug: Sample input values: {X_predict.iloc[0, :5].to_dict()}")

    print("Step 2: Making predictions with the trained model...")
    
    # Make predictions - these are absolute ARR values
    predicted_arr_values = model_pipeline.predict(X_predict)[0]
    print(f"🔍 Debug: Model predictions (absolute ARR) = {predicted_arr_values}")

    # --- Create forecast results ---
    print("Step 3: Creating forecast results...")
    future_quarters = ["FY26 Q1", "FY26 Q2", "FY26 Q3", "FY26 Q4"]
    forecast = []

    for i in range(4):
        predicted_arr = predicted_arr_values[i]
        
        # Calculate growth rate from current ARR
        current_arr = user_arr
        if i > 0:
            current_arr = predicted_arr_values[i-1]
        
        growth_rate = ((predicted_arr - current_arr) / current_arr) * 100 if current_arr > 0 else 0
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Quarterly Growth (%)": growth_rate
        })

    return pd.DataFrame(forecast)

def main():
    """Test the absolute ARR prediction with sample data."""
    print("🧪 TESTING ABSOLUTE ARR PREDICTION")
    print("=" * 50)
    
    # Load the model
    model_data = load_absolute_arr_model()
    if model_data is None:
        return
    
    # Create sample company data
    from enhanced_guided_input import EnhancedGuidedInputSystem
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    input_data = system.create_forecast_input_with_history(2800000, 800000)  # 40% quarterly growth
    
    # Make prediction
    result = predict_absolute_arr(model_data, input_data)
    
    print("\n🎯 PREDICTION RESULTS (ABSOLUTE ARR):")
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


