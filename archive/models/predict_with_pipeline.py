#!/usr/bin/env python3
"""
Uses the fully-contained scikit-learn pipeline to make a forecast.

This script loads the single pipeline.pkl file, which contains all
feature engineering, scaling, and model logic, and applies it to
a new company's raw historical data to generate a forecast.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
warnings.filterwarnings('ignore')

class FinancialDataTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform all feature engineering steps for the financial model.
    This ensures that the same logic is applied during training and inference.
    """
    def __init__(self):
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data, so fit is simple.
        return self

    def transform(self, X, y=None):
        """Applies all feature engineering transformations."""
        df_feat = X.copy()
        
        # Ensure data is sorted for temporal calculations
        if 'time_idx' in df_feat.columns and 'id_company' in df_feat.columns:
            df_feat = df_feat.sort_values(by=['id_company', 'time_idx'])

        # --- Base features to engineer from ---
        # We define this explicitly to avoid trying to create lags of lags, etc.
        base_metrics = [
            'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
            'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
        ]
        
        # --- Temporal Features (Lags & Rolling Windows) ---
        lags = [1, 2, 4]
        for col in base_metrics:
            if col not in df_feat.columns:
                continue
            for lag in lags:
                df_feat[f'{col}_lag_{lag}'] = df_feat.groupby('id_company')[col].shift(lag)
            
            df_feat[f'{col}_roll_mean_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
            df_feat[f'{col}_roll_std_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).std().shift(1))

        # --- Advanced SaaS Efficiency Metrics (Lagged to prevent leakage) ---
        # Only create these features if the required columns exist
        if 'Sales & Marketing' in df_feat.columns and 'Net New ARR' in df_feat.columns:
            sm_spend_lag2 = df_feat.groupby('id_company')['Sales & Marketing'].shift(2)
            nna_lag1 = df_feat.groupby('id_company')['Net New ARR'].shift(1)
            df_feat['Magic_Number_lag_1'] = nna_lag1 / sm_spend_lag2
        
        if 'Cash Burn (OCF & ICF)' in df_feat.columns and 'Net New ARR' in df_feat.columns:
            cash_burn_lag1 = df_feat.groupby('id_company')['Cash Burn (OCF & ICF)'].shift(1)
            nna_lag1 = df_feat.groupby('id_company')['Net New ARR'].shift(1)
            df_feat['Burn_Multiple_lag_1'] = np.abs(cash_burn_lag1) / nna_lag1
        
        # --- Other Growth & Ratio Features ---
        df_feat['HC_qoq_growth'] = df_feat.groupby('id_company')['Headcount (HC)'].pct_change(1)
        
        # --- Clean up generated features ---
        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # --- Final Feature Selection ---
        # Drop identifier columns and original base features that were used for engineering
        cols_to_drop = ['id_company', 'time_idx'] + base_metrics
        final_features = df_feat.drop(columns=[col for col in cols_to_drop if col in df_feat.columns])
        
        self.feature_names_out_ = final_features.columns.tolist()

        return final_features

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)

def main():
    """Main prediction function."""
    print("🚀 FORECASTING WITH THE UNIFIED PIPELINE")
    print("=" * 60)

    # 1. Load the unified pipeline
    print("Step 1: Loading the full pipeline from 'lightgbm_financial_model_absolute_arr.pkl'...")
    model_file = 'lightgbm_financial_model_absolute_arr.pkl'
    try:
        with open(model_file, 'rb') as f:
            pipeline = pickle.load(f)
        print("✅ Pipeline loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_file}'. Please run the training script first.")
        return

    # 2. Load and preprocess new company data (the raw inputs)
    print("\nStep 2: Loading and preparing raw data from 'test_company_2024.csv'...")
    try:
        new_data = pd.read_csv('test_company_2024.csv')
    except FileNotFoundError:
        print("❌ ERROR: 'test_company_2024.csv' not found.")
        return

    # --- Data Transformation (map to expected column names) ---
    new_data.rename(columns={
        'Quarter': 'Financial Quarter',
        'ARR_End_of_Quarter': 'cARR',
        'Quarterly_Net_New_ARR': 'Net New ARR',
        'Headcount': 'Headcount (HC)',
        'Gross_Margin_Percent': 'Gross Margin (in %)'
    }, inplace=True)
    
    # Add the necessary identifiers that the pipeline expects
    new_data['Year'] = new_data['Financial Quarter'].str.split(' ').str[1].astype(int)
    new_data['Quarter Num'] = new_data['Financial Quarter'].str.split(' ').str[0].str.replace('Q', '').astype(int)
    new_data['time_idx'] = new_data['Year'] * 4 + new_data['Quarter Num']
    new_data['id_company'] = 99999 # A dummy ID for this single company
    
    print("✅ Raw data transformed and identifiers added.")

    # 3. Make the prediction
    # The pipeline handles all feature engineering, scaling, and prediction internally.
    # We just need to provide the historical data for the last quarter.
    # The pipeline's transformer will use this data to create lags, etc.
    # Note: For a robust implementation, the pipeline would expect all historical data
    # to properly calculate rolling features. We pass the whole history.
    print("\nStep 3: Making the 4-quarter forecast...")
    predicted_log_arr = pipeline.predict(new_data)
    
    # We only care about the prediction from the last time step
    last_prediction_log = predicted_log_arr[-1]
    
    # Inverse transform the prediction from log scale back to absolute ARR
    predicted_arr = np.expm1(last_prediction_log)
    
    # 4. Display the results
    print("\n--- 📈 FORECAST RESULTS ---")
    last_known_arr = new_data['cARR'].iloc[-1]
    last_known_quarter = new_data['Financial Quarter'].iloc[-1]
    
    print(f"Last Known ARR ({last_known_quarter}): ${last_known_arr:,.0f}")
    print("-" * 30)
    
    year = new_data['Year'].iloc[-1]
    q_num = new_data['Quarter Num'].iloc[-1]
    
    for i, arr in enumerate(predicted_arr):
        next_q_num = (q_num + i) % 4 + 1
        next_year = year + (q_num + i) // 4
        print(f"Predicted ARR for FY{str(next_year)[2:]}Q{next_q_num}: ${arr:,.0f}")

    print("\n✅ Forecast complete.")


if __name__ == "__main__":
    main()
