#!/usr/bin/env python3
"""
Retrain the model to predict absolute ARR values instead of YoY growth rates.
This is what users actually want to see.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin

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
        sm_spend_lag2 = df_feat.groupby('id_company')['Sales & Marketing'].shift(2)
        nna_lag1 = df_feat.groupby('id_company')['Net New ARR'].shift(1)
        df_feat['Magic_Number_lag_1'] = nna_lag1 / sm_spend_lag2
        
        cash_burn_lag1 = df_feat.groupby('id_company')['Cash Burn (OCF & ICF)'].shift(1)
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

def load_and_clean_data(file_path):
    """Load and clean the corrected financial dataset."""
    print("Step 1: Loading and cleaning corrected data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading the data: {e}")
        return None

    # --- Time Index Creation ---
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)

    # --- Data Type Coercion ---
    potential_numeric_cols = df.columns.drop(['Financial Quarter', 'id_company'])
    for col in potential_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Nuanced Imputation Strategy ---
    df['Net New ARR'] = df.groupby('id_company')['cARR'].transform(lambda x: x.diff())
    
    # Forward-fill stock variables
    stock_vars = ['Headcount (HC)', 'Customers (EoP)']
    for col in stock_vars:
        if col in df.columns:
            df[col] = df.groupby('id_company')[col].transform(lambda x: x.ffill())
    
    # Fill flow variables with 0
    flow_vars = ['Net New ARR', 'Cash Burn (OCF & ICF)', 'Sales & Marketing', 'Expansion & Upsell', 'Churn & Reduction']
    for col in flow_vars:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Impute metrics with company-specific median
    df['Gross Margin (in %)'] = df.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
    # df['Gross Margin (in %)'].fillna(df['Gross Margin (in %)'].median(), inplace=True) # LEAK: Move to post-split

    print("✅ Data loading and cleaning complete.")
    return df

def create_multistep_targets(df, target_col='cARR', horizon=4):
    """Creates LOG-TRANSFORMED target variables for multi-step forecast."""
    print("Step 3: Creating multi-step LOG-TRANSFORMED target variables for absolute ARR...")
    df_target = df.copy()
    df_target = df_target.sort_values(by=['id_company', 'time_idx'])
    
    # We predict the log of ARR to stabilize variance and improve performance
    log_target_col = 'log_' + target_col
    df_target[log_target_col] = np.log1p(df_target[target_col])
    
    for i in range(1, horizon + 1):
        df_target[f'Target_Q{i}'] = df_target.groupby('id_company')[log_target_col].shift(-i)
    
    print("✅ Log-transformed target creation complete.")
    return df_target

def main():
    """Main training function."""
    print("🚀 RETRAINING MODEL TO PREDICT ABSOLUTE ARR VALUES")
    print("=" * 60)
    
    # Load corrected data
    df_clean = load_and_clean_data('202402_Copy_Fixed.csv')
    if df_clean is None:
        return
    
    # Engineer features
    df_featured = create_multistep_targets(df_clean, target_col='cARR', horizon=4)
    
    # Prepare for modeling
    target_cols = [f'Target_Q{i}' for i in range(1, 5)]
    df_model_ready = df_featured.copy()
    df_model_ready.dropna(subset=target_cols, inplace=True)
    
    # Define feature columns - X will now include id_company and time_idx for the transformer
    non_feature_cols = [
        'Financial Quarter', 'id', 'id_currency', 'id_sector', 'log_cARR',
        'id_target_customer', 'id_country', 'id_deal_team', 'Year', 'Quarter Num'
    ] + target_cols
    
    feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]
    
    X = df_model_ready[feature_cols]
    y = df_model_ready[target_cols]
    
    # Train-test split (on the data that includes identifiers)
    print("Step 4: Performing split by company...")
    company_ids = df_model_ready['id_company'].unique()
    train_cids, test_cids = train_test_split(company_ids, test_size=0.2, random_state=42)
    
    train_indices = df_model_ready[df_model_ready['id_company'].isin(train_cids)].index
    test_indices = df_model_ready[df_model_ready['id_company'].isin(test_cids)].index

    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    
    print(f" - Training set: {X_train.shape[0]} samples from {len(train_cids)} companies.")
    print(f" - Test set: {X_test.shape[0]} samples from {len(test_cids)} companies.")

    # --- Post-Split Data Imputation (to prevent leakage) ---
    # Impute missing values on the raw features before they enter the pipeline
    train_median = X_train.median()
    X_train = X_train.fillna(train_median)
    X_test = X_test.fillna(train_median)

    # --- Build the Full Scikit-learn Pipeline ---
    print("Step 5: Building the full feature engineering and modeling pipeline...")
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', FinancialDataTransformer()),
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(lgbm))
    ])

    # Train model
    print("Step 6: Training the full pipeline...")
    full_pipeline.fit(X_train, y_train)
    print("✅ Pipeline training complete.")
    
    # Evaluate model
    print("Step 7: Evaluating model performance...")
    y_pred_log = full_pipeline.predict(X_test)
    
    # Inverse transform predictions and actuals to original scale for interpretable metrics
    y_pred_arr = np.expm1(y_pred_log)
    y_test_arr = np.expm1(y_test.values) # Use .values to avoid index alignment issues
    
    y_pred_df = pd.DataFrame(y_pred_arr, columns=target_cols, index=y_test.index)
    
    print("\n--- 📈 MODEL PERFORMANCE RESULTS ---")
    # R² on the log scale is often a more stable and meaningful metric for skewed data
    overall_r2_log = r2_score(y_test, y_pred_log)
    print(f"Overall R² (on log-transformed scale): {overall_r2_log:.4f}")
    
    # R² on the original scale can be sensitive to outliers
    overall_r2_linear = r2_score(y_test_arr, y_pred_arr)
    print(f"Overall R² (on original ARR scale):   {overall_r2_linear:.4f}\n")
    
    results = {}
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test_arr[:, i], y_pred_arr[:, i])
        r2_linear = r2_score(y_test_arr[:, i], y_pred_arr[:, i])
        print(f"{col}: MAE = ${mae:,.0f}, R² (original scale) = {r2_linear:.4f}")
        results[col] = {'MAE': mae, 'R2_linear': r2_linear}
    
    # Save the entire pipeline
    import pickle
    with open('lightgbm_financial_model_absolute_arr.pkl', 'wb') as f:
        pickle.dump(full_pipeline, f)
    
    print(f"\n✅ Full model pipeline saved to lightgbm_financial_model_absolute_arr.pkl")
    
    # Feature importance analysis is more complex with a pipeline, so we'll simplify/skip for now
    print("\n✅ Model retraining with a unified pipeline complete!")

if __name__ == '__main__':
    main()

