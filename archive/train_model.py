#!/usr/bin/env python3
"""
Simple script to train the financial forecasting model.
Update the filepath variable below to point to your dataset.
"""

import sys
import os
from financial_forecasting_model import load_and_clean_data, engineer_features, create_multistep_targets, train_and_save_model

def main():
    """
    Main function to train the financial forecasting model.
    """
    print("🚀 Starting Financial Forecasting Model Training")
    print("=" * 60)
    
    # --------------------------------------------------------------------------
    # ❗ IMPORTANT: Update this path to your dataset file
    # --------------------------------------------------------------------------
    filepath = "202402_Copy.csv"  # ← Updated to use your uploaded dataset
    
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"❌ ERROR: Dataset file not found at '{filepath}'")
        print("Please update the 'filepath' variable in this script to point to your dataset.")
        print("\nExpected file format:")
        print("- CSV file with columns: Financial Quarter, id_company, cARR, ARR YoY Growth (in %), etc.")
        print("- Should contain historical financial data for multiple companies")
        return
    
    print(f"📁 Loading dataset from: {filepath}")
    
    # Step 1: Load and clean data
    df_clean = load_and_clean_data(filepath)
    
    if df_clean is None:
        print("❌ Failed to load and clean data. Please check your dataset.")
        return
    
    print(f"✅ Loaded {len(df_clean)} rows of data")
    print(f"✅ Found {df_clean['id_company'].nunique()} unique companies")
    
    # Step 2: Engineer features
    df_featured = engineer_features(df_clean)
    
    # Step 3: Create multi-step targets
    df_model_ready = create_multistep_targets(df_featured, target_col='ARR YoY Growth (in %)', horizon=4)
    
    # Step 4: Prepare final dataset
    target_cols = [f'Target_Q{i}' for i in range(1, 5)]
    df_model_ready.dropna(subset=target_cols, inplace=True)
    
    # Define feature columns
    non_feature_cols = ['Financial Quarter', 'id_company', 'time_idx', 'Year', 'Quarter Num', 'ARR YoY Growth (in %)'] + target_cols
    feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]
    
    print(f"✅ Created {len(feature_cols)} features")
    print(f"✅ Final dataset has {len(df_model_ready)} samples")
    
    # Fill any remaining NaNs
    df_model_ready[feature_cols] = df_model_ready[feature_cols].fillna(df_model_ready[feature_cols].median())
    
    # Step 5: Train and save model
    print("\n🎯 Starting model training...")
    model_data = train_and_save_model(df_model_ready, feature_cols, target_cols)
    
    if model_data:
        print("\n🎉 Training completed successfully!")
        print(f"📊 Model saved as: lightgbm_financial_model.pkl")
        print(f"📈 Overall R² Score: {model_data['overall_r2']:.4f}")
        print("\nYou can now use the prediction script to make forecasts on new data.")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 