#!/usr/bin/env python3
"""
Debug why the model is being so conservative with high-growth companies.
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

def analyze_training_data_growth():
    """Analyze the training data to understand growth patterns."""
    print("🔍 ANALYZING TRAINING DATA GROWTH PATTERNS")
    print("=" * 50)
    
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
    
    # Filter for realistic growth rates (remove extreme outliers)
    df_clean = df[(df['QoQ_Growth'] >= -50) & (df['QoQ_Growth'] <= 200)].copy()
    
    print(f"📊 Training Data Analysis:")
    print(f"Total quarters: {len(df_clean)}")
    print(f"Companies: {df_clean['id_company'].nunique()}")
    print(f"QoQ Growth Stats:")
    print(f"  Mean: {df_clean['QoQ_Growth'].mean():.2f}%")
    print(f"  Median: {df_clean['QoQ_Growth'].median():.2f}%")
    print(f"  75th percentile: {df_clean['QoQ_Growth'].quantile(0.75):.2f}%")
    print(f"  90th percentile: {df_clean['QoQ_Growth'].quantile(0.90):.2f}%")
    print(f"  95th percentile: {df_clean['QoQ_Growth'].quantile(0.95):.2f}%")
    print(f"  99th percentile: {df_clean['QoQ_Growth'].quantile(0.99):.2f}%")
    
    # Find companies with high growth
    high_growth_companies = df_clean[df_clean['QoQ_Growth'] >= 30]['id_company'].unique()
    print(f"\n🚀 Companies with 30%+ quarterly growth: {len(high_growth_companies)}")
    
    if len(high_growth_companies) > 0:
        print("Sample high-growth companies:")
        for company in high_growth_companies[:5]:
            company_data = df_clean[df_clean['id_company'] == company][['Financial Quarter', 'cARR', 'QoQ_Growth']].tail(4)
            print(f"Company {company}:")
            for _, row in company_data.iterrows():
                print(f"  {row['Financial Quarter']}: ${row['cARR']:,.0f} ({row['QoQ_Growth']:.1f}% growth)")
    
    return df_clean

def debug_test_company_features():
    """Debug what features the test company has vs what the model expects."""
    print("\n🔍 DEBUGGING TEST COMPANY FEATURES")
    print("=" * 50)
    
    # Load models
    model_data = load_single_quarter_models()
    if model_data is None:
        return
    
    # Prepare test company data (same as before)
    df = pd.read_csv('test_company_2024.csv')
    df_renamed = df.copy()
    df_renamed['Financial Quarter'] = df_renamed['Quarter']
    df_renamed['cARR'] = df_renamed['ARR_End_of_Quarter']
    df_renamed['Net New ARR'] = df_renamed['Quarterly_Net_New_ARR']
    df_renamed['Headcount (HC)'] = df_renamed['Headcount']
    df_renamed['Gross Margin (in %)'] = df_renamed['Gross_Margin_Percent']
    df_renamed['id_company'] = 'Test Company 2024'
    df_renamed['Revenue'] = df_renamed['cARR']
    df_renamed['ARR YoY Growth (in %)'] = 0
    df_renamed['Revenue YoY Growth (in %)'] = 0
    df_renamed['Cash Burn (OCF & ICF)'] = -df_renamed['cARR'] * 0.3
    df_renamed['Sales & Marketing'] = df_renamed['cARR'] * 0.2
    df_renamed['Expansion & Upsell'] = df_renamed['cARR'] * 0.1
    df_renamed['Churn & Reduction'] = -df_renamed['cARR'] * 0.05
    df_renamed['Customers (EoP)'] = df_renamed['Headcount (HC)'] * 10
    
    # Process the data
    df_clean = df_renamed.copy()
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})|(\d{4})')[0].fillna(df_clean['Financial Quarter'].str.extract(r'(\d{4})')[0]).astype(int)
    df_clean['Quarter Num'] = df_clean['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_clean['time_idx'] = df_clean['Year'] * 4 + df_clean['Quarter Num']
    df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Feature engineering
    processed_df = engineer_features(df_clean)
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # Check what features we have vs what the model expects
    feature_cols = model_data['feature_cols']
    missing_features = []
    present_features = []
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            present_features.append(col)
        else:
            missing_features.append(col)
    
    print(f"📊 Feature Analysis:")
    print(f"Total features expected: {len(feature_cols)}")
    print(f"Features present: {len(present_features)}")
    print(f"Features missing: {len(missing_features)}")
    
    print(f"\n✅ Key features present:")
    key_features = ['cARR', 'Net New ARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Sales & Marketing', 'Cash Burn (OCF & ICF)']
    for feature in key_features:
        if feature in present_features:
            value = prediction_input_row[feature].iloc[0]
            print(f"  {feature}: {value}")
    
    print(f"\n❌ Key features missing (using 0):")
    for feature in missing_features[:10]:  # Show first 10
        print(f"  {feature}: 0")
    if len(missing_features) > 10:
        print(f"  ... and {len(missing_features) - 10} more")
    
    # Calculate some key ratios that might be important
    current_arr = prediction_input_row['cARR'].iloc[0]
    net_new_arr = prediction_input_row['Net New ARR'].iloc[0]
    sm_spend = prediction_input_row['Sales & Marketing'].iloc[0]
    
    print(f"\n🔍 Key Ratios:")
    print(f"Net New ARR / ARR: {(net_new_arr/current_arr)*100:.1f}%")
    print(f"Magic Number: {net_new_arr/sm_spend if sm_spend > 0 else 'N/A'}")
    print(f"Growth Rate: {(net_new_arr/(current_arr-net_new_arr))*100:.1f}%")

def main():
    """Main debugging function."""
    print("🔍 DEBUGGING MODEL CONSERVATISM")
    print("=" * 60)
    
    # Analyze training data
    training_data = analyze_training_data_growth()
    
    # Debug test company features
    debug_test_company_features()
    
    print(f"\n💡 INSIGHTS:")
    print(f"1. The model was trained on companies with much lower growth rates")
    print(f"2. High-growth companies (30%+ quarterly) are rare in the training data")
    print(f"3. The model is predicting based on 'average' companies, not high-growth outliers")
    print(f"4. We need to either:")
    print(f"   - Retrain with more high-growth companies")
    print(f"   - Use a different approach for high-growth companies")
    print(f"   - Apply a growth momentum factor")

if __name__ == "__main__":
    main()


