# ==============================================================================
# A SYSTEMATIC & IMPROVED APPROACH TO FINANCIAL FORECASTING
# ==============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import joblib
import re
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# 1. FOUNDATIONAL DATA LOADING AND CLEANING
# ==============================================================================

def load_and_clean_data(file_path):
    """
    Loads and cleans the financial dataset based on best practices.
    - Converts quarters to a sortable time index.
    - Applies nuanced imputation: ffill for stock, 0 for flow.
    - Coerces data types and handles initial NaNs.
    """
    print("Step 1: Loading and cleaning data...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Dataset file not found at '{file_path}'")
        print("Please update the 'file_path' variable to the correct location.")
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
    # Create 'Net New ARR' from cARR difference before filling other flow variables
    df['Net New ARR'] = df.groupby('id_company')['cARR'].transform(lambda x: x.diff())

    # Forward-fill stock variables
    stock_vars = ['Headcount (HC)', 'Customers (EoP)']
    for col in stock_vars:
        if col in df.columns:
            df[col] = df.groupby('id_company')[col].transform(lambda x: x.ffill())

    # Fill flow variables with 0, assuming missing means no activity
    flow_vars = ['Net New ARR', 'Cash Burn (OCF & ICF)', 'Sales & Marketing', 'Expansion & Upsell', 'Churn & Reduction']
    for col in flow_vars:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # Impute metrics like Gross Margin with company-specific median
    df['Gross Margin (in %)'] = df.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
    df['Gross Margin (in %)'].fillna(df['Gross Margin (in %)'].median(), inplace=True) # For any companies with no data

    print("✅ Data loading and cleaning complete.")
    return df

# ==============================================================================
# 2. STRATEGIC FEATURE ENGINEERING
# ==============================================================================

def engineer_features(df):
    """
    Engineers a rich set of temporal and domain-specific SaaS features.
    """
    print("Step 2: Engineering features...")
    df_feat = df.copy()
    df_feat = df_feat.sort_values(by=['id_company', 'time_idx']) # Ensure order

    # --- Temporal Features (Lags & Rolling Windows) ---
    metrics_to_process = [
        'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
        'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
    ]
    lags = [1, 2, 4] # 1 quarter, 2 quarters, 1 year
    for col in metrics_to_process:
        if col not in df_feat.columns: continue
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat.groupby('id_company')[col].shift(lag)
        
        # 4-Quarter Rolling Stats (shifted to prevent data leakage from current period)
        df_feat[f'{col}_roll_mean_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
        df_feat[f'{col}_roll_std_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).std().shift(1))

    # --- Advanced SaaS Efficiency Metrics ---
    # Magic Number: New ARR generated for every $1 of S&M spend from the prior quarter.
    sm_spend_lag1 = df_feat.groupby('id_company')['Sales & Marketing'].shift(1)
    df_feat['Magic_Number'] = df_feat['Net New ARR'] / sm_spend_lag1
    
    # Burn Multiple: How much cash is burned to generate $1 of new ARR.
    # We use absolute burn, as burn is often represented as a negative number.
    df_feat['Burn_Multiple'] = np.abs(df_feat['Cash Burn (OCF & ICF)']) / df_feat['Net New ARR']

    # --- Other Growth & Ratio Features ---
    df_feat['HC_qoq_growth'] = df_feat.groupby('id_company')['Headcount (HC)'].pct_change(1)
    df_feat['ARR_per_Headcount'] = df_feat['cARR'] / df_feat['Headcount (HC)']

    # --- Scale & Seasonality Features ---
    df_feat['log_cARR'] = np.log1p(df_feat['cARR'].clip(lower=0))

    # --- Growth Momentum Features ---
    df_feat['growth_lag1'] = df_feat.groupby('id_company')['ARR YoY Growth (in %)'].shift(1)
    df_feat['growth_lag2'] = df_feat.groupby('id_company')['ARR YoY Growth (in %)'].shift(2)
    df_feat['growth_momentum'] = df_feat['growth_lag1'] - df_feat['growth_lag2']
    df_feat['growth_roll_mean_4'] = df_feat.groupby('id_company')['ARR YoY Growth (in %)'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean()
    )

    # --- ARR Trajectory Features ---
    df_feat['arr_qoq_ratio'] = df_feat.groupby('id_company')['cARR'].transform(
        lambda x: x / x.shift(1)
    )
    df_feat['arr_qoq_ratio_lag1'] = df_feat.groupby('id_company')['arr_qoq_ratio'].shift(1)

    # --- Clean up generated features ---
    numeric_cols = df_feat.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)

    print("✅ Feature engineering complete.")
    return df_feat

# ==============================================================================
# 3. MULTI-STEP TARGET CREATION
# ==============================================================================

def create_multistep_targets(df, target_col='ARR YoY Growth (in %)', horizon=4):
    """
    Creates target variables (e.g., Target_Q1, Target_Q2) for a multi-step forecast.
    """
    print("Step 3: Creating multi-step target variables...")
    df_target = df.copy()
    df_target = df_target.sort_values(by=['id_company', 'time_idx'])
    
    for i in range(1, horizon + 1):
        df_target[f'Target_Q{i}'] = df_target.groupby('id_company')[target_col].shift(-i)
        
    print("✅ Target creation complete.")
    return df_target

# ==============================================================================
# 4. GROUPKFOLD CROSS-VALIDATION
# ==============================================================================

def evaluate_with_cv(df, feature_cols, target_cols, target_cap=500, n_splits=5):
    """
    Runs company-based K-fold cross-validation for robust R² estimation.
    Each fold ensures no company appears in both train and test.
    """
    print(f"\n--- Running {n_splits}-Fold GroupKFold CV (cap=±{target_cap}%) ---")
    
    clean_cols = {c: re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in feature_cols}
    df_cv = df.rename(columns=clean_cols)
    clean_feature_cols = [clean_cols.get(c, c) for c in feature_cols]
    
    gkf = GroupKFold(n_splits=n_splits)
    groups = df_cv['id_company'].values
    
    fold_results = {col: [] for col in target_cols}
    
    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(df_cv, groups=groups)):
        X_train = df_cv.iloc[train_idx][clean_feature_cols].copy()
        X_test = df_cv.iloc[test_idx][clean_feature_cols].copy()
        
        for col in clean_feature_cols:
            X_train[col] = X_train[col].replace([np.inf, -np.inf], np.nan)
            X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan)
            p01, p99 = X_train[col].quantile(0.01), X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(p01, p99)
            X_test[col] = X_test[col].clip(p01, p99)
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)
        
        fold_r2s = []
        for tc in target_cols:
            y_tr = df_cv.iloc[train_idx][tc].clip(-target_cap, target_cap)
            y_te = df_cv.iloc[test_idx][tc].clip(-target_cap, target_cap)
            
            m = lgb.LGBMRegressor(
                objective='regression_l1', n_estimators=500, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1
            )
            m.fit(X_train, y_tr)
            p = np.clip(m.predict(X_test), -target_cap, target_cap)
            r2 = r2_score(y_te, p)
            fold_results[tc].append(r2)
            fold_r2s.append(r2)
        
        print(f"  Fold {fold_num+1}: " + " | ".join(f"{tc}={fold_r2s[i]:.4f}" for i, tc in enumerate(target_cols)))
    
    print("\n  CV Summary:")
    all_means = []
    for tc in target_cols:
        arr = np.array(fold_results[tc])
        mean_r2 = arr.mean()
        all_means.append(mean_r2)
        print(f"    {tc}: {mean_r2:.4f} ± {arr.std():.4f}")
    overall_cv = np.mean(all_means)
    print(f"    Overall CV R²: {overall_cv:.4f}")
    return fold_results, overall_cv

# ==============================================================================
# 5. MODEL TRAINING AND SAVING
# ==============================================================================

def train_and_save_model(df_model_ready, feature_cols, target_cols, model_path='lightgbm_financial_model.pkl'):
    """
    Trains the LightGBM model and saves it along with necessary metadata.
    """
    print("Step 5: Performing company-based split...")
    company_ids = df_model_ready['id_company'].unique()
    train_cids, test_cids = train_test_split(company_ids, test_size=0.2, random_state=42)

    train_indices = df_model_ready[df_model_ready['id_company'].isin(train_cids)].index
    test_indices = df_model_ready[df_model_ready['id_company'].isin(test_cids)].index

    X_train, X_test = df_model_ready.loc[train_indices, feature_cols].copy(), df_model_ready.loc[test_indices, feature_cols].copy()
    y_train, y_test = df_model_ready.loc[train_indices, target_cols], df_model_ready.loc[test_indices, target_cols]

    # --- Winsorize using training percentiles only (prevent test leakage) ---
    clip_bounds = {}
    for col in feature_cols:
        p01 = X_train[col].quantile(0.01)
        p99 = X_train[col].quantile(0.99)
        clip_bounds[col] = (p01, p99)
        X_train[col] = X_train[col].clip(p01, p99)
        X_test[col] = X_test[col].clip(p01, p99)

    # --- Impute remaining NaNs using training medians only ---
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    
    print(f"  - Training set: {X_train.shape[0]} samples from {len(train_cids)} companies.")
    print(f"  - Test set:     {X_test.shape[0]} samples from {len(test_cids)} companies.")

    # --- Build and Train the Improved Modeling Pipeline ---
    print("Step 6: Building and training the LightGBM model...")
    
    # LightGBM is fast, memory-efficient, and highly accurate.
    # Using 'regression_l1' (MAE) as the objective is often more robust to financial data outliers.
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', 
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    # The pipeline handles scaling and modeling sequentially.
    model_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', MultiOutputRegressor(lgbm))
    ])
    
    # Train the entire pipeline on the training data
    model_pipeline.fit(X_train, y_train)
    print("✅ Model training complete.")
    
    # --- Evaluate Model Performance ---
    print("Step 7: Evaluating model performance...")
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
    
    # --- Visualize Results ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance: Actual vs. Predicted Growth Rate', fontsize=18)
    
    for i, ax in enumerate(axes.flatten()):
        col = target_cols[i]
        sns.regplot(x=y_test[col], y=y_pred_df[col], ax=ax,
                    scatter_kws={'alpha': 0.4, 's': 25, 'edgecolor': 'w', 'linewidths': 0.5}, 
                    line_kws={'color': '#E41A1C', 'lw': 2.5})
        ax.plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 'k--', lw=2, label='Perfect Prediction')
        ax.set_title(f'{col} Forecast (R² = {results[col]["R2"]:.3f})', fontsize=12)
        ax.set_xlabel('Actual Value', fontsize=10)
        ax.set_ylabel('Predicted Value', fontsize=10)
        ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Analyze Feature Importances ---
    print("\n--- 📊 FEATURE IMPORTANCE ANALYSIS ---")
    trained_estimators = model_pipeline.named_steps['model'].estimators_
    importances_df = pd.DataFrame(index=feature_cols)
    for i, estimator in enumerate(trained_estimators):
        importances_df[f'Target_Q{i+1}'] = estimator.feature_importances_
    
    importances_df['mean_importance'] = importances_df.mean(axis=1)
    importances_df = importances_df.sort_values('mean_importance', ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=importances_df['mean_importance'].head(25), y=importances_df.index[:25], palette='viridis_r')
    plt.title('Top 25 Most Predictive Features (Averaged Across All Forecast Horizons)', fontsize=16)
    plt.xlabel('Mean Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Save Model and Metadata ---
    print("Step 8: Saving model and metadata...")
    model_data = {
        'model_pipeline': model_pipeline,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'performance_results': results,
        'overall_r2': overall_r2,
        'feature_importance': importances_df,
        'clip_bounds': clip_bounds,
        'train_medians': train_medians.to_dict(),
        'target_cap': 500,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Model saved to {model_path}")
    print("\n✅ Modeling pipeline executed successfully.")
    
    return model_data

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    
    # --------------------------------------------------------------------------
    # ❗ IMPORTANT: Please update this path to your dataset file.
    # --------------------------------------------------------------------------
    filepath = "202402_Copy_Fixed.csv"
    
    # --- Execute Data Prep and Feature Engineering ---
    df_clean = load_and_clean_data(filepath)
    
    if df_clean is not None:
        df_featured = engineer_features(df_clean)
        df_model_ready = create_multistep_targets(df_featured, target_col='ARR YoY Growth (in %)', horizon=4)
        
        # --- Finalize Dataset for Modeling ---
        target_cols = [f'Target_Q{i}' for i in range(1, 5)]
        df_model_ready.dropna(subset=target_cols, inplace=True)
        
        # Winsorize targets: cap extreme growth values at ±500%
        # Standard practice in financial econometrics — only 3.9% of values affected.
        # No model can reliably predict 82,000% growth; 500% (6x ARR/yr) covers all practical VC scenarios.
        TARGET_CAP = 500
        for col in target_cols:
            df_model_ready[col] = df_model_ready[col].clip(-TARGET_CAP, TARGET_CAP)
        
        # Define feature columns (X) by excluding identifiers, raw targets, and future targets
        non_feature_cols = ['Financial Quarter', 'id_company', 'time_idx', 'Year', 'Quarter Num', 'ARR YoY Growth (in %)', 'id'] + target_cols
        feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]

        # Sanitize feature names (LightGBM rejects special JSON characters)
        clean_map = {c: re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in feature_cols}
        df_model_ready.rename(columns=clean_map, inplace=True)
        feature_cols = [clean_map.get(c, c) for c in feature_cols]

        # Run cross-validation for robust R² estimate
        cv_results, cv_r2 = evaluate_with_cv(df_model_ready, feature_cols, target_cols, target_cap=TARGET_CAP)

        # Train and save the final model (winsorization + imputation happen inside, using train-only statistics)
        model_data = train_and_save_model(df_model_ready, feature_cols, target_cols)