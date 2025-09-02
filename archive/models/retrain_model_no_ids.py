#!/usr/bin/env python3
"""
Retrain the LightGBM model without problematic ID features that cause overfitting.
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
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def load_and_clean_data(file_path):
    """Loads and cleans the financial dataset."""
    print("Step 1: Loading and cleaning data...")
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

    # Impute metrics like Gross Margin with company-specific median
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

def create_multistep_targets(df, target_col='ARR YoY Growth (in %)', horizon=4):
    """Creates target variables for multi-step forecast."""
    print("Step 3: Creating multi-step target variables...")
    df_target = df.copy()
    df_target = df_target.sort_values(by=['id_company', 'time_idx'])
    
    for i in range(1, horizon + 1):
        df_target[f'Target_Q{i}'] = df_target.groupby('id_company')[target_col].shift(-i)
        
    print("✅ Target creation complete.")
    return df_target

def train_and_save_model(df_model_ready, feature_cols, target_cols, model_path='lightgbm_financial_model.pkl'):
    """Trains the LightGBM model and saves it."""
    print("Step 4: Performing temporal split by company...")
    company_ids = df_model_ready['id_company'].unique()
    train_cids, test_cids = train_test_split(company_ids, test_size=0.2, random_state=42)

    train_indices = df_model_ready['id_company'].isin(train_cids).index
    test_indices = df_model_ready[df_model_ready['id_company'].isin(test_cids)].index

    X_train, X_test = df_model_ready.loc[train_indices, feature_cols], df_model_ready.loc[test_indices, feature_cols]
    y_train, y_test = df_model_ready.loc[train_indices, target_cols], df_model_ready.loc[test_indices, target_cols]
    
    print(f"  - Training set: {X_train.shape[0]} samples from {len(train_cids)} companies.")
    print(f"  - Test set:     {X_test.shape[0]} samples from {len(test_cids)} companies.")

    # --- Build and Train the Model ---
    print("Step 5: Building and training the LightGBM model...")
    
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

    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(lgbm))
    ])
    
    model_pipeline.fit(X_train, y_train)
    print("✅ Model training complete.")
    
    # --- Evaluate Model Performance ---
    print("Step 6: Evaluating model performance...")
    y_pred = model_pipeline.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

    print("\n--- 📈 MODEL PERFORMANCE RESULTS ---")
    overall_r2 = r2_score(y_test, y_pred)
    print(f"Overall R² (across all horizons): {overall_r2:.4f}\n")
    
    results = {}
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test[col], y_pred[:, i])
        r2 = r2_score(y_test[col], y_pred[:, i])
        print(f"{col}: MAE = {mae:.4f}, R² = {r2:.4f}")
        results[col] = {'MAE': mae, 'R2': r2}
    
    # --- Analyze Feature Importances ---
    print("\n--- 📊 FEATURE IMPORTANCE ANALYSIS ---")
    trained_estimators = model_pipeline.named_steps['model'].estimators_
    importances_df = pd.DataFrame(index=feature_cols)
    for i, estimator in enumerate(trained_estimators):
        importances_df[f'Target_Q{i+1}'] = estimator.feature_importances_
    
    importances_df['mean_importance'] = importances_df.mean(axis=1)
    importances_df = importances_df.sort_values('mean_importance', ascending=False)
    
    print("Top 20 most important features:")
    for i, (feature, row) in enumerate(importances_df.head(20).iterrows()):
        print(f"{i+1:2d}. {feature}: {row['mean_importance']:.4f}")
    
    # --- Save Model and Metadata ---
    print("Step 7: Saving model and metadata...")
    model_data = {
        'model_pipeline': model_pipeline,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'performance_results': results,
        'overall_r2': overall_r2,
        'feature_importance': importances_df
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {model_path}")
    print("\n✅ Modeling pipeline executed successfully.")
    
    return model_data

def main():
    """Main execution function."""
    print("🚀 RETRAINING MODEL WITHOUT ID FEATURES")
    print("=" * 50)
    
    # Load and process data
    df_clean = load_and_clean_data('202402_Copy.csv')
    
    if df_clean is not None:
        df_featured = engineer_features(df_clean)
        df_model_ready = create_multistep_targets(df_featured, target_col='ARR YoY Growth (in %)', horizon=4)
        
        # --- Finalize Dataset for Modeling ---
        target_cols = [f'Target_Q{i}' for i in range(1, 5)]
        df_model_ready.dropna(subset=target_cols, inplace=True)
        
        # Define feature columns - EXCLUDE ALL ID FEATURES
        non_feature_cols = [
            'Financial Quarter', 'id_company', 'time_idx', 'Year', 'Quarter Num', 
            'ARR YoY Growth (in %)', 'id', 'id_currency', 'id_sector', 
            'id_target_customer', 'id_country', 'id_deal_team'
        ] + target_cols
        
        feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]
        
        print(f"🔍 Excluded {len(non_feature_cols)} non-feature columns")
        print(f"🔍 Using {len(feature_cols)} feature columns")
        print(f"🔍 Excluded ID features: {[col for col in non_feature_cols if 'id' in col]}")
        
        # Fill any remaining NaNs
        df_model_ready[feature_cols] = df_model_ready[feature_cols].fillna(df_model_ready[feature_cols].median())
        
        # Train and save the model
        model_data = train_and_save_model(df_model_ready, feature_cols, target_cols)
        
        print("\n🎉 RETRAINING COMPLETE!")
        print("The new model should perform much better on new companies.")
        
    else:
        print("❌ Cannot proceed without data. Please check the dataset path.")

if __name__ == '__main__':
    main()


