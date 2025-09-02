#!/usr/bin/env python3
"""
Debug why the bias correction is making every Q1 prediction 11.8%.
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

def debug_bias_correction():
    """Debug the bias correction process."""
    print("🔍 DEBUGGING BIAS CORRECTION")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Test with a few different companies
    test_cases = [
        {"name": "High-Growth", "arr": 5000000, "net_new": 1500000},
        {"name": "Moderate-Growth", "arr": 10000000, "net_new": 2000000},
        {"name": "Low-Growth", "arr": 50000000, "net_new": 5000000}
    ]
    
    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"🏢 TESTING: {case['name']} Company")
        print(f"{'='*50}")
        
        # Create test data
        system = EnhancedGuidedInputSystem()
        system.initialize_from_training_data()
        input_data = system.create_forecast_input_with_history(case["arr"], case["net_new"])
        
        # Process data
        df_renamed = input_data.copy()
        df_renamed['Year'] = df_renamed['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
        df_renamed['Year'] = df_renamed['Year'].apply(lambda x: x + 2000 if x < 100 else x)
        df_renamed['Quarter Num'] = df_renamed['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
        df_renamed['time_idx'] = df_renamed['Year'] * 4 + df_renamed['Quarter Num']
        df_renamed = df_renamed.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
        
        # Ensure numeric columns are numeric
        numeric_cols = df_renamed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce')
        
        # Feature engineering
        processed_df = engineer_features(df_renamed)
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
        
        # Get predictions
        print("📊 Original YoY Predictions:")
        yoy_predictions = []
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            model = models[quarter]
            predicted_yoy_growth = model.predict(X_predict)[0]
            yoy_predictions.append(predicted_yoy_growth)
            print(f"  {quarter}: {predicted_yoy_growth:.1f}%")
        
        # Apply bias correction
        correction_factors = [0.8, 1.1, 1.1, 1.0]
        corrected_yoy = [yoy * factor for yoy, factor in zip(yoy_predictions, correction_factors)]
        
        print(f"\n📊 Corrected YoY Predictions:")
        for i, (original, corrected) in enumerate(zip(yoy_predictions, corrected_yoy)):
            quarter = f"Q{i+1}"
            print(f"  {quarter}: {original:.1f}% → {corrected:.1f}%")
        
        # Convert to quarterly growth
        print(f"\n📊 Quarterly Growth Rates:")
        for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            quarterly_growth_rate = ((1 + corrected_yoy[i]/100) ** (1/4) - 1) * 100
            print(f"  {quarter}: {quarterly_growth_rate:.1f}%")
        
        # Show some key input features
        print(f"\n📊 Key Input Features:")
        key_features = ['cARR', 'Net New ARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Sales & Marketing']
        for feature in key_features:
            if feature in X_predict.columns:
                value = X_predict[feature].iloc[0]
                print(f"  {feature}: {value}")

def analyze_correction_formula():
    """Analyze the correction formula to see why it's producing similar results."""
    print(f"\n🔍 ANALYZING CORRECTION FORMULA")
    print("=" * 60)
    
    # Test different YoY growth rates
    test_yoy_rates = [50, 100, 150, 200, 300]
    
    print("📊 Correction Formula Analysis:")
    print(f"{'Original YoY':<12} {'Q1 Factor':<10} {'Corrected YoY':<15} {'Quarterly Growth':<15}")
    print("-" * 60)
    
    for yoy_rate in test_yoy_rates:
        correction_factor = 0.8  # Q1 factor
        corrected_yoy = yoy_rate * correction_factor
        quarterly_growth = ((1 + corrected_yoy/100) ** (1/4) - 1) * 100
        
        print(f"{yoy_rate:>10.0f}% {correction_factor:>8.1f} {corrected_yoy:>13.1f}% {quarterly_growth:>13.1f}%")
    
    print(f"\n💡 INSIGHT: The correction formula is too rigid!")
    print(f"All YoY rates get multiplied by 0.8, then converted to quarterly growth.")
    print(f"This means similar YoY rates will always produce similar quarterly rates.")

def main():
    """Main debugging function."""
    print("🔍 DEBUGGING BIAS CORRECTION ISSUE")
    print("=" * 80)
    
    # Debug bias correction
    debug_bias_correction()
    
    # Analyze correction formula
    analyze_correction_formula()
    
    print(f"\n💡 CONCLUSIONS:")
    print(f"1. The bias correction is too rigid - it applies the same factors to all companies")
    print(f"2. All YoY predictions get multiplied by 0.8 for Q1, leading to similar quarterly rates")
    print(f"3. We need a more sophisticated bias correction that considers company characteristics")
    print(f"4. The correction should be relative to the company's growth profile, not absolute")

if __name__ == "__main__":
    main()


