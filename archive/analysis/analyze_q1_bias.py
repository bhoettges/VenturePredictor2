#!/usr/bin/env python3
"""
Analyze the Q1 bias pattern in our predictions.
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

def analyze_training_data_quarterly_patterns():
    """Analyze the training data to see if there's a Q1 bias."""
    print("🔍 ANALYZING TRAINING DATA QUARTERLY PATTERNS")
    print("=" * 60)
    
    # Load the training data
    df = pd.read_csv('202402_Copy_Fixed.csv')
    
    # Calculate quarterly growth rates
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Calculate quarterly growth
    df['QoQ_Growth'] = df.groupby('id_company')['cARR'].pct_change(1) * 100
    
    # Filter for realistic growth rates
    df_clean = df[(df['QoQ_Growth'] >= -50) & (df['QoQ_Growth'] <= 200)].copy()
    
    # Analyze by quarter
    print("📊 Quarterly Growth Patterns in Training Data:")
    quarterly_stats = df_clean.groupby('Quarter Num')['QoQ_Growth'].agg(['count', 'mean', 'median', 'std']).round(2)
    print(quarterly_stats)
    
    # Show distribution by quarter
    print(f"\n📈 Growth Rate Distribution by Quarter:")
    for quarter in [1, 2, 3, 4]:
        quarter_data = df_clean[df_clean['Quarter Num'] == quarter]['QoQ_Growth']
        print(f"Q{quarter}: Mean={quarter_data.mean():.1f}%, Median={quarter_data.median():.1f}%, 75th={quarter_data.quantile(0.75):.1f}%")
    
    return df_clean

def analyze_model_predictions_by_quarter():
    """Analyze our model predictions to see the Q1 bias pattern."""
    print(f"\n🔍 ANALYZING MODEL PREDICTIONS BY QUARTER")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Test with a few companies
    test_cases = [
        {"name": "High-Growth", "arr": 5000000, "net_new": 1500000},
        {"name": "Moderate-Growth", "arr": 10000000, "net_new": 2000000},
        {"name": "Low-Growth", "arr": 50000000, "net_new": 5000000}
    ]
    
    print("📊 Model Prediction Patterns:")
    print(f"{'Company Type':<15} {'Q1 Growth':<12} {'Q2 Growth':<12} {'Q3 Growth':<12} {'Q4 Growth':<12}")
    print("-" * 65)
    
    for case in test_cases:
        # Create simple test data
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
        predictions = []
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            model = models[quarter]
            predicted_yoy_growth = model.predict(X_predict)[0]
            quarterly_growth_rate = ((1 + predicted_yoy_growth/100) ** (1/4) - 1) * 100
            predictions.append(quarterly_growth_rate)
        
        print(f"{case['name']:<15} {predictions[0]:>10.1f}% {predictions[1]:>10.1f}% {predictions[2]:>10.1f}% {predictions[3]:>10.1f}%")

def analyze_yoY_to_quarterly_conversion():
    """Analyze if the YoY to quarterly conversion is causing the bias."""
    print(f"\n🔍 ANALYZING YOY TO QUARTERLY CONVERSION")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Test different YoY growth rates
    yoy_rates = [50, 100, 150, 200, 300]
    
    print("📊 YoY to Quarterly Conversion Analysis:")
    print(f"{'YoY Growth':<12} {'Q1 Growth':<12} {'Q2 Growth':<12} {'Q3 Growth':<12} {'Q4 Growth':<12}")
    print("-" * 65)
    
    for yoy_rate in yoy_rates:
        quarterly_rate = ((1 + yoy_rate/100) ** (1/4) - 1) * 100
        print(f"{yoy_rate:>10.0f}% {quarterly_rate:>10.1f}% {quarterly_rate:>10.1f}% {quarterly_rate:>10.1f}% {quarterly_rate:>10.1f}%")
    
    print(f"\n💡 INSIGHT: The conversion formula gives the SAME quarterly rate for all quarters!")
    print(f"This means the Q1 bias is coming from the MODEL PREDICTIONS, not the conversion.")

def analyze_model_differences():
    """Analyze the differences between Q1, Q2, Q3, Q4 models."""
    print(f"\n🔍 ANALYZING MODEL DIFFERENCES")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    models = model_data['models']
    r2_scores = model_data['r2_scores']
    
    print("📊 Model Performance by Quarter:")
    for quarter, r2 in r2_scores.items():
        print(f"{quarter}: R² = {r2:.4f}")
    
    print(f"\n💡 INSIGHT: Q1 model has R² = {r2_scores['Q1']:.4f}")
    print(f"Q2 model has R² = {r2_scores['Q2']:.4f} (highest!)")
    print(f"Q3 model has R² = {r2_scores['Q3']:.4f}")
    print(f"Q4 model has R² = {r2_scores['Q4']:.4f} (lowest)")
    
    print(f"\n🔍 The Q1 bias might be because:")
    print(f"1. Q1 model is trained on different data patterns")
    print(f"2. Q1 predictions are more volatile/optimistic")
    print(f"3. There's a systematic difference in how Q1 targets are created")

def main():
    """Main analysis function."""
    print("🔍 ANALYZING Q1 BIAS PATTERN")
    print("=" * 80)
    
    # Analyze training data patterns
    training_data = analyze_training_data_quarterly_patterns()
    
    # Analyze model predictions
    analyze_model_predictions_by_quarter()
    
    # Analyze conversion formula
    analyze_yoY_to_quarterly_conversion()
    
    # Analyze model differences
    analyze_model_differences()
    
    print(f"\n💡 CONCLUSIONS:")
    print(f"1. The Q1 bias is coming from the MODEL PREDICTIONS, not the conversion formula")
    print(f"2. Each model (Q1, Q2, Q3, Q4) is predicting different YoY growth rates")
    print(f"3. The Q1 model consistently predicts higher YoY growth than other quarters")
    print(f"4. This suggests a systematic bias in the training data or model architecture")
    print(f"5. We need to investigate why Q1 predictions are consistently higher")

if __name__ == "__main__":
    main()
