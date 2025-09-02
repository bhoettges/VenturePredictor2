#!/usr/bin/env python3
"""
Predict quarterly growth rates directly using the quarterly growth model.
This gives users exactly what they want: quarterly ARR progression.
"""

import pandas as pd
import numpy as np
import pickle
from financial_forecasting_model import load_and_clean_data, engineer_features

def load_quarterly_growth_model():
    """Load the model trained to predict quarterly growth rates."""
    try:
        with open('lightgbm_financial_model_quarterly_growth.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("✅ Model loaded successfully from lightgbm_financial_model_quarterly_growth.pkl")
        return model_data
    except FileNotFoundError:
        print("❌ ERROR: Model file not found. Please run retrain_for_quarterly_growth.py first.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load model: {e}")
        return None

def predict_quarterly_arr_progression(model_data, company_df):
    """
    Predicts quarterly ARR progression using quarterly growth rates.
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

    print("Step 2: Making predictions with the trained model...")
    
    # Make predictions - these are quarterly growth rates
    predicted_growth_rates = model_pipeline.predict(X_predict)[0]
    print(f"🔍 Debug: Model predictions (quarterly growth rates) = {predicted_growth_rates}")

    # --- Create forecast results ---
    print("Step 3: Creating quarterly ARR progression...")
    last_known_q = processed_df.iloc[-1]
    current_arr = last_known_q['cARR']
    future_quarters = ["FY26 Q1", "FY26 Q2", "FY26 Q3", "FY26 Q4"]
    forecast = []

    print(f"🔍 Starting ARR: ${current_arr:,.0f}")
    
    for i in range(4):
        quarterly_growth_rate = predicted_growth_rates[i] / 100  # Convert percentage to decimal
        
        # Apply quarterly growth to get next quarter's ARR
        predicted_arr = current_arr * (1 + quarterly_growth_rate)
        
        # Calculate uncertainty bounds (±10%)
        uncertainty_factor = 0.10
        lower_bound = predicted_arr * (1 - uncertainty_factor)
        upper_bound = predicted_arr * (1 + uncertainty_factor)
        
        print(f"🔍 Q{i+1} Debug: Growth Rate = {predicted_growth_rates[i]:.1f}% → ARR = ${predicted_arr:,.0f}")
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound,
            "Quarterly Growth (%)": predicted_growth_rates[i]
        })
        
        # Update current ARR for next iteration
        current_arr = predicted_arr

    return pd.DataFrame(forecast)

def main():
    """Test the quarterly growth prediction with sample data."""
    print("🧪 TESTING QUARTERLY GROWTH PREDICTION")
    print("=" * 50)
    
    # Load the model
    model_data = load_quarterly_growth_model()
    if model_data is None:
        return
    
    # Create sample company data
    from enhanced_guided_input import EnhancedGuidedInputSystem
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    input_data = system.create_forecast_input_with_history(2800000, 800000)  # 40% quarterly growth
    
    # Make prediction
    result = predict_quarterly_arr_progression(model_data, input_data)
    
    print("\n🎯 PREDICTION RESULTS (QUARTERLY GROWTH):")
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


