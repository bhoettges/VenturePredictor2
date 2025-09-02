#!/usr/bin/env python3
"""
Simple investigation into the Q1 bias issue.
"""

import pandas as pd
import numpy as np
import pickle

def load_single_quarter_models():
    """Load the single-quarter models."""
    try:
        with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return None

def analyze_training_data_simple():
    """Simple analysis of training data quarterly patterns."""
    print("🔍 ANALYZING TRAINING DATA QUARTERLY PATTERNS")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv('202402_Copy.csv')
    
    # Add quarter information
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    # Analyze quarterly patterns
    print("📊 QUARTERLY GROWTH PATTERNS IN TRAINING DATA:")
    print("-" * 60)
    
    quarterly_stats = df.groupby('Quarter Num')['ARR YoY Growth (in %)'].agg(['mean', 'median', 'std', 'count']).round(2)
    print(quarterly_stats)
    
    # Check if Q1 has higher growth rates
    q1_mean = quarterly_stats.loc[1, 'mean']
    q2_mean = quarterly_stats.loc[2, 'mean']
    q3_mean = quarterly_stats.loc[3, 'mean']
    q4_mean = quarterly_stats.loc[4, 'mean']
    
    print(f"\n💡 INSIGHT:")
    print(f"Q1 Mean Growth: {q1_mean:.2f}%")
    print(f"Q2 Mean Growth: {q2_mean:.2f}%")
    print(f"Q3 Mean Growth: {q3_mean:.2f}%")
    print(f"Q4 Mean Growth: {q4_mean:.2f}%")
    
    if q1_mean > q2_mean and q1_mean > q3_mean and q1_mean > q4_mean:
        print("🚨 Q1 HAS HIGHEST GROWTH IN TRAINING DATA!")
        print("This explains why Q1 predictions are higher!")
    else:
        print("🤔 Q1 doesn't have the highest growth in training data...")
    
    return df

def test_model_predictions_simple():
    """Test what the models actually predict."""
    print(f"\n🔍 TESTING MODEL PREDICTIONS")
    print("=" * 60)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    models = model_data['models']
    feature_cols = model_data['feature_cols']
    
    # Create simple test data
    print("📊 Testing with simple company data...")
    
    # Create a simple test case
    test_arr = 5000000
    test_net_new = 1000000
    
    # Create minimal input data
    from enhanced_guided_input import EnhancedGuidedInputSystem
    system = EnhancedGuidedInputSystem()
    system.initialize_from_training_data()
    
    # Create input data
    input_data = system.create_forecast_input_with_history(test_arr, test_net_new)
    
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
    from financial_forecasting_model import engineer_features
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
    yoy_predictions = []
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        model = models[quarter]
        predicted_yoy = model.predict(X_predict)[0]
        yoy_predictions.append(predicted_yoy)
        print(f"  {quarter} Model: {predicted_yoy:.1f}% YoY")
    
    # Check if Q1 model predicts higher YoY
    if yoy_predictions[0] > yoy_predictions[1] and yoy_predictions[0] > yoy_predictions[2] and yoy_predictions[0] > yoy_predictions[3]:
        print("🚨 Q1 MODEL PREDICTS HIGHEST YOY GROWTH!")
        print("This is the source of the Q1 bias!")
    else:
        print("🤔 Q1 model doesn't predict the highest YoY...")
    
    # Convert to quarterly growth
    print("\n📊 Converted to Quarterly Growth:")
    quarterly_predictions = []
    for i, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        predicted_yoy = yoy_predictions[i]
        quarterly_growth = ((1 + predicted_yoy/100) ** (1/4) - 1) * 100
        quarterly_predictions.append(quarterly_growth)
        print(f"  {quarter}: {quarterly_growth:.1f}% quarterly")
    
    # Check if Q1 has highest quarterly growth
    if quarterly_predictions[0] > quarterly_predictions[1] and quarterly_predictions[0] > quarterly_predictions[2] and quarterly_predictions[0] > quarterly_predictions[3]:
        print("🚨 Q1 HAS HIGHEST QUARTERLY GROWTH!")
        print("This confirms the Q1 bias!")
    else:
        print("🤔 Q1 doesn't have the highest quarterly growth...")

def analyze_training_targets_simple():
    """Analyze what each quarter model was trained to predict."""
    print(f"\n🔍 ANALYZING TRAINING TARGETS")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv('202402_Copy.csv')
    
    # Add quarter information
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    print("📊 What Each Quarter Model Was Trained To Predict:")
    print("-" * 60)
    
    for quarter_num in [1, 2, 3, 4]:
        quarter_name = f"Q{quarter_num}"
        
        # Get the target values for this quarter
        quarter_data = df[df['Quarter Num'] == quarter_num]
        target_values = quarter_data['ARR YoY Growth (in %)']
        
        print(f"\n{quarter_name} Model Training Targets:")
        print(f"  Count: {len(target_values)}")
        print(f"  Mean: {target_values.mean():.2f}%")
        print(f"  Median: {target_values.median():.2f}%")
        print(f"  Std: {target_values.std():.2f}%")
    
    # Check if Q1 has higher targets
    q1_targets = df[df['Quarter Num'] == 1]['ARR YoY Growth (in %)']
    q2_targets = df[df['Quarter Num'] == 2]['ARR YoY Growth (in %)']
    q3_targets = df[df['Quarter Num'] == 3]['ARR YoY Growth (in %)']
    q4_targets = df[df['Quarter Num'] == 4]['ARR YoY Growth (in %)']
    
    print(f"\n💡 TARGET COMPARISON:")
    print(f"Q1 Mean Target: {q1_targets.mean():.2f}%")
    print(f"Q2 Mean Target: {q2_targets.mean():.2f}%")
    print(f"Q3 Mean Target: {q3_targets.mean():.2f}%")
    print(f"Q4 Mean Target: {q4_targets.mean():.2f}%")
    
    if q1_targets.mean() > q2_targets.mean() and q1_targets.mean() > q3_targets.mean() and q1_targets.mean() > q4_targets.mean():
        print("🚨 Q1 HAS HIGHEST TARGET VALUES IN TRAINING DATA!")
        print("This is why the Q1 model predicts higher values!")
    else:
        print("🤔 Q1 doesn't have the highest target values...")

def main():
    """Main investigation function."""
    print("🔍 SIMPLE INVESTIGATION INTO Q1 BIAS ISSUE")
    print("=" * 80)
    
    # 1. Analyze training data patterns
    df = analyze_training_data_simple()
    
    # 2. Test model predictions
    test_model_predictions_simple()
    
    # 3. Analyze training targets
    analyze_training_targets_simple()
    
    print(f"\n{'='*80}")
    print("🎯 INVESTIGATION SUMMARY")
    print(f"{'='*80}")
    print("The investigation should reveal:")
    print("1. Whether Q1 has higher growth in training data")
    print("2. Whether Q1 model predicts higher YoY values")
    print("3. Whether Q1 has higher target values")
    print("\nThis will explain the Q1 bias!")

if __name__ == "__main__":
    main()


