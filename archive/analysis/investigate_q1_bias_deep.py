#!/usr/bin/env python3
"""
Deep investigation into the Q1 bias issue.
Let's understand why Q1 predictions are consistently higher.
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
        return model_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return None

def analyze_training_data_quarterly_patterns():
    """Analyze the training data to understand quarterly patterns."""
    print("🔍 ANALYZING TRAINING DATA QUARTERLY PATTERNS")
    print("=" * 60)
    
    # Load training data
    df = load_and_clean_data('202402_Copy.csv')
    if df is None:
        return
    
    # Add quarter information
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    # Calculate Net New ARR
    df['Net New ARR'] = df.groupby('id_company')['cARR'].transform(lambda x: x.diff())
    
    # Analyze quarterly patterns
    print("📊 QUARTERLY GROWTH PATTERNS IN TRAINING DATA:")
    print("-" * 60)
    
    quarterly_stats = df.groupby('Quarter Num').agg({
        'ARR YoY Growth (in %)': ['mean', 'median', 'std', 'count'],
        'Net New ARR': ['mean', 'median', 'std'],
        'cARR': ['mean', 'median']
    }).round(2)
    
    print(quarterly_stats)
    
    # Look at specific examples
    print(f"\n📊 SAMPLE COMPANIES BY QUARTER:")
    print("-" * 60)
    
    # Get a few companies with data across all quarters
    companies_with_all_quarters = df.groupby('id_company')['Quarter Num'].nunique()
    complete_companies = companies_with_all_quarters[companies_with_all_quarters >= 4].index[:5]
    
    for company in complete_companies:
        company_data = df[df['id_company'] == company].sort_values('Quarter Num')
        print(f"\n🏢 {company}:")
        for _, row in company_data.iterrows():
            print(f"  Q{row['Quarter Num']}: ARR=${row['cARR']:,.0f}, YoY={row['ARR YoY Growth (in %)']:.1f}%")
    
    return df

def analyze_model_predictions_by_quarter():
    """Analyze what the models are actually predicting for each quarter."""
    print(f"\n🔍 ANALYZING MODEL PREDICTIONS BY QUARTER")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Create test data for different company types
    test_cases = [
        {"name": "Small Company", "arr": 2000000, "net_new": 400000},
        {"name": "Medium Company", "arr": 8000000, "net_new": 1600000},
        {"name": "Large Company", "arr": 20000000, "net_new": 3000000}
    ]
    
    for case in test_cases:
        print(f"\n🏢 TESTING: {case['name']}")
        print("-" * 40)
        
        # Create simple test data
        from enhanced_guided_input import EnhancedGuidedInputSystem
        system = EnhancedGuidedInputSystem()
        system.initialize_from_training_data()
        
        # Create input data
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
        X_predict = pd.DataFrame(index=prediction_input_row.index, columns=feature_cols)
        for col in feature_cols:
            if col in prediction_input_row.columns:
                X_predict[col] = prediction_input_row[col]
            else:
                X_predict[col] = 0
        X_predict = X_predict.fillna(0)
        
        # Get predictions from each quarter model
        print("📊 Raw YoY Predictions by Quarter Model:")
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            model = models[quarter]
            predicted_yoy = model.predict(X_predict)[0]
            print(f"  {quarter} Model: {predicted_yoy:.1f}% YoY")
        
        # Convert to quarterly growth
        print("📊 Converted to Quarterly Growth:")
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            model = models[quarter]
            predicted_yoy = model.predict(X_predict)[0]
            quarterly_growth = ((1 + predicted_yoy/100) ** (1/4) - 1) * 100
            print(f"  {quarter}: {quarterly_growth:.1f}% quarterly")

def analyze_feature_importance_by_quarter():
    """Analyze feature importance differences between quarter models."""
    print(f"\n🔍 ANALYZING FEATURE IMPORTANCE BY QUARTER")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Get feature importance for each quarter model
    print("📊 Top 10 Features by Quarter Model:")
    print("-" * 60)
    
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        model = models[quarter]
        importances = model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n{quarter} Model Top Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

def analyze_training_targets():
    """Analyze the actual targets used for training each quarter model."""
    print(f"\n🔍 ANALYZING TRAINING TARGETS BY QUARTER")
    print("=" * 60)
    
    # Load training data
    df = load_and_clean_data('202402_Copy.csv')
    if df is None:
        return
    
    # Add quarter information
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    # Create targets for each quarter (what each model was trained to predict)
    df_sorted = df.sort_values(by=['id_company', 'Year', 'Quarter Num'])
    
    # For each quarter model, what was it trained to predict?
    print("📊 What Each Quarter Model Was Trained To Predict:")
    print("-" * 60)
    
    for quarter_num in [1, 2, 3, 4]:
        quarter_name = f"Q{quarter_num}"
        
        # Get the target values for this quarter
        quarter_data = df_sorted[df_sorted['Quarter Num'] == quarter_num]
        target_values = quarter_data['ARR YoY Growth (in %)']
        
        print(f"\n{quarter_name} Model Training Targets:")
        print(f"  Count: {len(target_values)}")
        print(f"  Mean: {target_values.mean():.2f}%")
        print(f"  Median: {target_values.median():.2f}%")
        print(f"  Std: {target_values.std():.2f}%")
        print(f"  Min: {target_values.min():.2f}%")
        print(f"  Max: {target_values.max():.2f}%")
        
        # Show distribution
        print(f"  Distribution:")
        for percentile in [25, 50, 75, 90, 95]:
            value = target_values.quantile(percentile/100)
            print(f"    {percentile}th percentile: {value:.1f}%")

def investigate_conversion_logic():
    """Investigate the YoY to quarterly conversion logic."""
    print(f"\n🔍 INVESTIGATING YOY TO QUARTERLY CONVERSION LOGIC")
    print("=" * 60)
    
    # Test different YoY values and see what quarterly growth they produce
    test_yoy_values = [50, 100, 150, 200, 300]
    
    print("📊 YoY to Quarterly Conversion Test:")
    print("-" * 60)
    print(f"{'YoY Growth':<12} {'Quarterly Growth':<15} {'Formula':<30}")
    print("-" * 60)
    
    for yoy in test_yoy_values:
        # Current conversion formula: ((1 + yoy/100) ** (1/4) - 1) * 100
        quarterly = ((1 + yoy/100) ** (1/4) - 1) * 100
        formula = f"(1 + {yoy}/100)^(1/4) - 1"
        print(f"{yoy:>10}% {quarterly:>13.1f}% {formula:<30}")
    
    print(f"\n💡 INSIGHT: The conversion formula assumes compound quarterly growth!")
    print(f"This means if YoY growth is 100%, each quarter grows by ~18.9%")
    print(f"After 4 quarters: (1.189)^4 = 2.0 = 100% YoY growth")

def main():
    """Main investigation function."""
    print("🔍 DEEP INVESTIGATION INTO Q1 BIAS ISSUE")
    print("=" * 80)
    
    # 1. Analyze training data patterns
    df = analyze_training_data_quarterly_patterns()
    
    # 2. Analyze model predictions
    analyze_model_predictions_by_quarter()
    
    # 3. Analyze feature importance
    analyze_feature_importance_by_quarter()
    
    # 4. Analyze training targets
    analyze_training_targets()
    
    # 5. Investigate conversion logic
    investigate_conversion_logic()
    
    print(f"\n{'='*80}")
    print("🎯 INVESTIGATION SUMMARY")
    print(f"{'='*80}")
    print("1. Check if training data has Q1 bias")
    print("2. Check if models predict different YoY values for Q1")
    print("3. Check if feature importance differs by quarter")
    print("4. Check if training targets differ by quarter")
    print("5. Check if conversion logic is correct")
    print("\nThis should reveal the root cause of the Q1 bias!")

if __name__ == "__main__":
    main()


