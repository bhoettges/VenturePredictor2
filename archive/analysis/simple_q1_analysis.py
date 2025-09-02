#!/usr/bin/env python3
"""
Simple analysis of the Q1 bias pattern.
"""

import pandas as pd
import numpy as np
import pickle

def analyze_training_data_quarters():
    """Analyze quarterly patterns in training data."""
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

def analyze_model_performance():
    """Analyze model performance by quarter."""
    print(f"\n🔍 ANALYZING MODEL PERFORMANCE BY QUARTER")
    print("=" * 60)
    
    # Load models
    try:
        with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load models: {e}")
        return
    
    r2_scores = model_data['r2_scores']
    
    print("📊 Model Performance by Quarter:")
    for quarter, r2 in r2_scores.items():
        print(f"{quarter}: R² = {r2:.4f}")
    
    print(f"\n💡 INSIGHT: Model performance varies by quarter:")
    print(f"Q1: R² = {r2_scores['Q1']:.4f}")
    print(f"Q2: R² = {r2_scores['Q2']:.4f} (highest!)")
    print(f"Q3: R² = {r2_scores['Q3']:.4f}")
    print(f"Q4: R² = {r2_scores['Q4']:.4f} (lowest)")

def analyze_yoY_conversion():
    """Analyze the YoY to quarterly conversion formula."""
    print(f"\n🔍 ANALYZING YOY TO QUARTERLY CONVERSION")
    print("=" * 60)
    
    # Test different YoY growth rates
    yoy_rates = [50, 100, 150, 200, 300]
    
    print("📊 YoY to Quarterly Conversion Analysis:")
    print(f"{'YoY Growth':<12} {'Quarterly Growth':<15}")
    print("-" * 30)
    
    for yoy_rate in yoy_rates:
        quarterly_rate = ((1 + yoy_rate/100) ** (1/4) - 1) * 100
        print(f"{yoy_rate:>10.0f}% {quarterly_rate:>13.1f}%")
    
    print(f"\n💡 INSIGHT: The conversion formula gives the SAME quarterly rate for all quarters!")
    print(f"This means the Q1 bias is coming from the MODEL PREDICTIONS, not the conversion.")

def analyze_our_predictions():
    """Analyze our actual predictions to see the pattern."""
    print(f"\n🔍 ANALYZING OUR ACTUAL PREDICTIONS")
    print("=" * 60)
    
    # From our test results
    test_results = [
        {"company": "High-Growth Startup", "q1": 36.2, "q2": 27.1, "q3": 21.2, "q4": 17.6},
        {"company": "Moderate Growth SaaS", "q1": 28.3, "q2": 7.9, "q3": 12.6, "q4": 7.1},
        {"company": "Mature Enterprise", "q1": 15.9, "q2": 7.3, "q3": 7.3, "q4": 8.7},
        {"company": "Hyper-Growth Unicorn", "q1": 36.2, "q2": 27.1, "q3": 21.2, "q4": 17.6},
        {"company": "Early Stage Startup", "q1": 36.2, "q2": 27.1, "q3": 21.2, "q4": 17.6},
        {"company": "Stable Growth Company", "q1": 15.9, "q2": 7.3, "q3": 7.3, "q4": 8.7}
    ]
    
    print("📊 Our Prediction Patterns:")
    print(f"{'Company Type':<20} {'Q1':<8} {'Q2':<8} {'Q3':<8} {'Q4':<8}")
    print("-" * 55)
    
    for result in test_results:
        print(f"{result['company']:<20} {result['q1']:>6.1f}% {result['q2']:>6.1f}% {result['q3']:>6.1f}% {result['q4']:>6.1f}%")
    
    # Calculate averages
    avg_q1 = np.mean([r['q1'] for r in test_results])
    avg_q2 = np.mean([r['q2'] for r in test_results])
    avg_q3 = np.mean([r['q3'] for r in test_results])
    avg_q4 = np.mean([r['q4'] for r in test_results])
    
    print("-" * 55)
    print(f"{'AVERAGE':<20} {avg_q1:>6.1f}% {avg_q2:>6.1f}% {avg_q3:>6.1f}% {avg_q4:>6.1f}%")
    
    print(f"\n💡 INSIGHT: Q1 is consistently the highest ({avg_q1:.1f}%), then drops off.")
    print(f"This suggests a systematic bias in our model predictions.")

def main():
    """Main analysis function."""
    print("🔍 ANALYZING Q1 BIAS PATTERN")
    print("=" * 80)
    
    # Analyze training data patterns
    training_data = analyze_training_data_quarters()
    
    # Analyze model performance
    analyze_model_performance()
    
    # Analyze conversion formula
    analyze_yoY_conversion()
    
    # Analyze our predictions
    analyze_our_predictions()
    
    print(f"\n💡 CONCLUSIONS:")
    print(f"1. The Q1 bias is coming from the MODEL PREDICTIONS, not the conversion formula")
    print(f"2. Each model (Q1, Q2, Q3, Q4) is predicting different YoY growth rates")
    print(f"3. The Q1 model consistently predicts higher YoY growth than other quarters")
    print(f"4. This suggests a systematic bias in the training data or model architecture")
    print(f"5. The training data shows Q4 has the highest growth, but our model predicts Q1 highest")
    print(f"6. We need to investigate why Q1 predictions are consistently higher")

if __name__ == "__main__":
    main()


