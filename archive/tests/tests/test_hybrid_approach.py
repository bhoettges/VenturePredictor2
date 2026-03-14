#!/usr/bin/env python3
"""
Test a hybrid approach: Use high-accuracy YoY model but post-process to eliminate Q1 bias.
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

def predict_with_bias_correction(model_data, company_df):
    """
    Predict using high-accuracy YoY model but correct for Q1 bias.
    """
    print("Step 1: Making predictions with high-accuracy YoY model...")
    
    # Load the trained models
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Process the company data (same as before)
    df_clean = company_df.copy()
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df_clean['Year'] = df_clean['Year'].apply(lambda x: x + 2000 if x < 100 else x)
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
    print(f"🔍 Starting ARR: ${current_arr:,.0f}")
    
    # Make predictions with YoY model
    yoy_predictions = []
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        model = models[quarter]
        predicted_yoy_growth = model.predict(X_predict)[0]
        yoy_predictions.append(predicted_yoy_growth)
        print(f"🔍 {quarter} YoY Prediction: {predicted_yoy_growth:.1f}%")
    
    # Step 2: Apply bias correction
    print("\nStep 2: Applying bias correction...")
    
    # Calculate the average YoY prediction
    avg_yoy = np.mean(yoy_predictions)
    print(f"🔍 Average YoY Prediction: {avg_yoy:.1f}%")
    
    # Apply bias correction: reduce Q1 bias, increase Q2-Q4 predictions
    correction_factors = [0.8, 1.1, 1.1, 1.0]  # Reduce Q1, increase Q2-Q3, keep Q4
    corrected_yoy = [yoy * factor for yoy, factor in zip(yoy_predictions, correction_factors)]
    
    print(f"🔍 Corrected YoY Predictions:")
    for i, (original, corrected) in enumerate(zip(yoy_predictions, corrected_yoy)):
        quarter = f"Q{i+1}"
        print(f"  {quarter}: {original:.1f}% → {corrected:.1f}%")
    
    # Step 3: Convert to quarterly growth and create forecast
    print("\nStep 3: Creating quarterly ARR progression...")
    
    future_quarters = ["FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4"]
    forecast = []
    
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        # Convert corrected YoY growth to quarterly growth rate
        quarterly_growth_rate = ((1 + corrected_yoy[i]/100) ** (1/4) - 1) * 100
        
        # Apply quarterly growth to get next quarter's ARR
        predicted_arr = current_arr * (1 + quarterly_growth_rate/100)
        
        # Calculate uncertainty bounds (±10%)
        uncertainty_factor = 0.10
        lower_bound = predicted_arr * (1 - uncertainty_factor)
        upper_bound = predicted_arr * (1 + uncertainty_factor)
        
        print(f"🔍 {quarter} Debug: YoY Growth = {corrected_yoy[i]:.1f}% → QoQ Growth = {quarterly_growth_rate:.1f}% → ARR = ${predicted_arr:,.0f}")
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound,
            "Quarterly Growth (%)": quarterly_growth_rate,
            "YoY Growth (%)": corrected_yoy[i]
        })
        
        # Update current ARR for next iteration
        current_arr = predicted_arr

    return pd.DataFrame(forecast)

def main():
    """Test the hybrid approach with bias correction."""
    print("🧪 TESTING HYBRID APPROACH WITH BIAS CORRECTION")
    print("=" * 60)
    
    # Load the high-accuracy models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Show model performance
    print("\n📊 MODEL PERFORMANCE:")
    for quarter, r2 in model_data['r2_scores'].items():
        print(f"{quarter}: R² = {r2:.4f}")
    print(f"Overall R²: {np.mean(list(model_data['r2_scores'].values())):.4f}")
    
    # Create test company data
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    input_data = system.create_forecast_input_with_history(2800000, 800000)  # 40% quarterly growth
    
    # Make prediction with bias correction
    result = predict_with_bias_correction(model_data, input_data)
    
    print("\n🎯 PREDICTION RESULTS WITH BIAS CORRECTION:")
    print("=" * 50)
    print(result)
    
    print("\n📊 SUMMARY:")
    current_arr = 2800000
    final_arr = result.iloc[-1]['Predicted ARR ($)']
    total_growth = ((final_arr - current_arr) / current_arr) * 100
    print(f"Current ARR: ${current_arr:,.0f}")
    print(f"Final ARR: ${final_arr:,.0f}")
    print(f"Total Growth: {total_growth:.1f}% over 4 quarters")
    
    # Show quarterly growth pattern
    print(f"\n📈 QUARTERLY GROWTH PATTERN:")
    for _, row in result.iterrows():
        print(f"{row['Future Quarter']}: {row['Quarterly Growth (%)']:.1f}% growth → ${row['Predicted ARR ($)']:,.0f}")

if __name__ == "__main__":
    main()


