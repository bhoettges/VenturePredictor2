#!/usr/bin/env python3
"""
Train separate models for each quarter (Q1, Q2, Q3, Q4) to achieve much higher accuracy.
Each model only predicts 1 quarter ahead, which should be much more accurate.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
    df['Gross Margin (in %)'].fillna(df['Gross Margin (in %)'].median(), inplace=True)

    print("✅ Data loading and cleaning complete.")
    return df

def engineer_features(df):
    """Engineers features without ID-based overfitting."""
    print("Step 2: Engineering features...")
    df_feat = df.copy()
    df_feat = df_feat.sort_values(by=['id_company', 'time_idx'])

    # --- Temporal Features (Lags & Rolling Windows) ---
    metrics_to_process = [
        'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
        'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
    ]
    lags = [1, 2, 4]
    for col in metrics_to_process:
        if col not in df_feat.columns: 
            continue
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat.groupby('id_company')[col].shift(lag)
        
        # 4-Quarter Rolling Stats
        df_feat[f'{col}_roll_mean_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
        df_feat[f'{col}_roll_std_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).std().shift(1))

    # --- Advanced SaaS Efficiency Metrics ---
    sm_spend_lag1 = df_feat.groupby('id_company')['Sales & Marketing'].shift(1)
    df_feat['Magic_Number'] = df_feat['Net New ARR'] / sm_spend_lag1
    df_feat['Burn_Multiple'] = np.abs(df_feat['Cash Burn (OCF & ICF)']) / df_feat['Net New ARR']

    # --- Other Growth & Ratio Features ---
    df_feat['HC_qoq_growth'] = df_feat.groupby('id_company')['Headcount (HC)'].pct_change(1)
    df_feat['ARR_per_Headcount'] = df_feat['cARR'] / df_feat['Headcount (HC)']
    
    # --- Clean up generated features ---
    numeric_cols = df_feat.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
        p01 = df_feat[col].quantile(0.01)
        p99 = df_feat[col].quantile(0.99)
        df_feat[col] = df_feat[col].clip(p01, p99)

    print("✅ Feature engineering complete.")
    return df_feat

def create_single_quarter_targets(df, target_col='ARR YoY Growth (in %)', horizon=4):
    """Creates single-quarter targets for each model."""
    print("Step 3: Creating single-quarter target variables...")
    df_target = df.copy()
    df_target = df_target.sort_values(by=['id_company', 'time_idx'])
    
    # Create targets for each quarter ahead
    for i in range(1, horizon + 1):
        df_target[f'Target_Q{i}'] = df_target.groupby('id_company')[target_col].shift(-i)
    
    print("✅ Single-quarter target creation complete.")
    return df_target

def train_single_quarter_model(X_train, y_train, X_test, y_test, quarter_name, feature_cols):
    """Train a single model for one quarter prediction."""
    print(f"Training {quarter_name} model...")
    
    # LightGBM configuration
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
    
    # Pipeline with scaling
    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', lgbm)
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ {quarter_name} Model: MAE = {mae:.4f}, R² = {r2:.4f}")
    
    return model_pipeline, r2

def main():
    """Main training function for single-quarter models."""
    print("🚀 TRAINING SINGLE-QUARTER PREDICTION MODELS")
    print("=" * 60)
    
    # Load corrected data
    df_clean = load_and_clean_data('202402_Copy_Fixed.csv')
    if df_clean is None:
        return
    
    # Engineer features
    df_featured = engineer_features(df_clean)
    df_model_ready = create_single_quarter_targets(df_featured, target_col='ARR YoY Growth (in %)', horizon=4)
    
    # Prepare for modeling
    target_cols = [f'Target_Q{i}' for i in range(1, 5)]
    df_model_ready.dropna(subset=target_cols, inplace=True)
    
    # Define feature columns (excluding ID features and targets)
    non_feature_cols = [
        'Financial Quarter', 'id_company', 'time_idx', 'Year', 'Quarter Num', 
        'cARR', 'id', 'id_currency', 'id_sector', 
        'id_target_customer', 'id_country', 'id_deal_team'
    ] + target_cols
    feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]
    
    print(f"🔍 Using {len(feature_cols)} feature columns")
    print(f"🔍 Excluded ID features: {[col for col in non_feature_cols if 'id' in col or 'company' in col]}")
    
    X = df_model_ready[feature_cols]
    y = df_model_ready[target_cols]
    
    # Fill any remaining NaNs
    X = X.fillna(X.median())
    
    # Train-test split
    print("Step 4: Performing temporal split by company...")
    company_ids = df_model_ready['id_company'].unique()
    train_cids, test_cids = train_test_split(company_ids, test_size=0.2, random_state=42)
    train_indices = df_model_ready['id_company'].isin(train_cids).index
    test_indices = df_model_ready[df_model_ready['id_company'].isin(test_cids)].index
    X_train, X_test = X.loc[train_indices], X.loc[test_indices]
    y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    
    print(f" - Training set: {X_train.shape[0]} samples from {len(train_cids)} companies.")
    print(f" - Test set: {X_test.shape[0]} samples from {len(test_cids)} companies.")
    
    # Train separate models for each quarter
    print("Step 5: Training separate models for each quarter...")
    models = {}
    r2_scores = {}
    
    for i, target_col in enumerate(target_cols):
        quarter_name = f"Q{i+1}"
        y_train_quarter = y_train[target_col]
        y_test_quarter = y_test[target_col]
        
        # Remove rows where target is NaN
        valid_train = ~y_train_quarter.isna()
        valid_test = ~y_test_quarter.isna()
        
        X_train_clean = X_train[valid_train]
        y_train_clean = y_train_quarter[valid_train]
        X_test_clean = X_test[valid_test]
        y_test_clean = y_test_quarter[valid_test]
        
        print(f"\n--- Training {quarter_name} Model ---")
        print(f"Training samples: {len(X_train_clean)}")
        print(f"Test samples: {len(X_test_clean)}")
        
        model, r2 = train_single_quarter_model(
            X_train_clean, y_train_clean, X_test_clean, y_test_clean, 
            quarter_name, feature_cols
        )
        
        models[quarter_name] = model
        r2_scores[quarter_name] = r2
    
    # Save all models
    import pickle
    model_data = {
        'models': models,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'r2_scores': r2_scores
    }
    
    with open('lightgbm_single_quarter_models.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ All models saved to lightgbm_single_quarter_models.pkl")
    
    # Summary
    print("\n--- 📈 SINGLE-QUARTER MODEL PERFORMANCE ---")
    overall_r2 = np.mean(list(r2_scores.values()))
    print(f"Overall R² (averaged): {overall_r2:.4f}")
    print()
    for quarter, r2 in r2_scores.items():
        print(f"{quarter}: R² = {r2:.4f}")
    
    if overall_r2 >= 0.70:
        print(f"\n🎉 SUCCESS! Overall R² = {overall_r2:.4f} is production-ready!")
    else:
        print(f"\n⚠️  Overall R² = {overall_r2:.4f} is still below 70% threshold")
    
    print("\n✅ Single-quarter model training complete!")

if __name__ == '__main__':
    main()


