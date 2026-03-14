# ==============================================================================
# FINANCIAL FORECASTING PREDICTION SCRIPT
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from financial_forecasting_model import load_and_clean_data, engineer_features

warnings.filterwarnings('ignore')

def load_trained_model(model_path='lightgbm_financial_model.pkl'):
    """
    Loads the trained model and metadata from the pickle file.
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"✅ Model loaded successfully from {model_path}")
        return model_data
    except FileNotFoundError:
        print(f"❌ ERROR: Model file not found at '{model_path}'")
        print("Please ensure the model has been trained and saved first.")
        return None
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred while loading the model: {e}")
        return None

def engineer_features(df):
    """
    Engineers a rich set of temporal and domain-specific SaaS features.
    (This MUST be identical to the function used for training the model)
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
    
    # --- Clean up generated features ---
    numeric_cols = df_feat.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        # Replace infinities from divisions with NaN
        df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
        # Winsorize extreme outliers to the 1st and 99th percentiles
        p01 = df_feat[col].quantile(0.01)
        p99 = df_feat[col].quantile(0.99)
        df_feat[col] = df_feat[col].clip(p01, p99)

    print("✅ Feature engineering complete.")
    return df_feat

def predict_future_arr(model_data, company_df):
    """
    Predicts future ARR growth rates for a company using the trained model.
    """
    print("Step 1: Loading and preprocessing company data...")
    
    # Load the trained model
    model_pipeline = model_data['model_pipeline']
    feature_cols = model_data['feature_cols']
    
    # Process the company data directly (since we already have a DataFrame)
    print("Step 1.1: Processing company data...")
    
    # Apply the same cleaning logic as in load_and_clean_data
    df_clean = company_df.copy()
    
    # --- Time Index Creation (must match training exactly) ---
    df_clean['Year'] = df_clean['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df_clean['Year'] = df_clean['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df_clean['Quarter Num'] = df_clean['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_clean['time_idx'] = df_clean['Year'] * 4 + df_clean['Quarter Num']
    df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)

    # --- Data Type Coercion ---
    potential_numeric_cols = df_clean.columns.drop(['Financial Quarter', 'id_company'])
    for col in potential_numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # --- Nuanced Imputation Strategy (must match training exactly) ---
    # Create 'Net New ARR' from cARR difference before filling other flow variables
    df_clean['Net New ARR'] = df_clean.groupby('id_company')['cARR'].transform(lambda x: x.diff())

    # Forward-fill stock variables
    stock_vars = ['Headcount (HC)', 'Customers (EoP)']
    for col in stock_vars:
        if col in df_clean.columns:
            df_clean[col] = df_clean.groupby('id_company')[col].transform(lambda x: x.ffill())

    # Fill flow variables with 0, assuming missing means no activity
    flow_vars = ['Net New ARR', 'Cash Burn (OCF & ICF)', 'Sales & Marketing', 'Expansion & Upsell', 'Churn & Reduction']
    for col in flow_vars:
        if col in df_clean.columns:
            df_clean[col].fillna(0, inplace=True)

    # Impute metrics like Gross Margin with company-specific median
    df_clean['Gross Margin (in %)'] = df_clean.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
    df_clean['Gross Margin (in %)'].fillna(df_clean['Gross Margin (in %)'].median(), inplace=True) # For any companies with no data
    
    # Process the company data (same as training)
    processed_df = engineer_features(df_clean)
    
    # Get the most recent quarter for prediction
    prediction_input_row = processed_df.iloc[-1:].copy()
    
    # Create feature matrix for prediction
    X_predict = pd.DataFrame(index=prediction_input_row.index)
    
    # Smart feature imputation (using enhanced_guided_input defaults)
    print("Step 1.5: Using smart defaults from enhanced_guided_input...")
    
    # The enhanced_guided_input system already provides smart defaults
    # We just need to ensure all required features are present
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            # Use 0 for any missing features (enhanced_guided_input should have provided all needed features)
            X_predict[col] = 0
            print(f"⚠️  Warning: Missing feature '{col}', using 0")

    # Fill any remaining NaNs with 0
    X_predict = X_predict.fillna(0)
    
    print(f"🔍 Debug: Input features shape: {X_predict.shape}")
    print(f"🔍 Debug: Sample input values: {X_predict.iloc[0, :5].to_dict()}")

    print("Step 2: Making predictions with the trained model...")
    
    # Check if this is an extreme growth company that needs fallback
    user_arr = prediction_input_row['cARR'].iloc[0] if 'cARR' in prediction_input_row.columns else 1000000
    user_growth = prediction_input_row['ARR YoY Growth (in %)'].iloc[0] if 'ARR YoY Growth (in %)' in prediction_input_row.columns else 20
    
    # Use the trained model directly (no more fallback needed with corrected data)
    predicted_growth_rates = model_pipeline.predict(X_predict)[0]
    print(f"🔍 Debug: Model predictions = {predicted_growth_rates}")

    # --- Translate YoY growth rates into quarterly ARR progression ---
    print("Step 3: Calculating quarterly ARR progression using proper YoY logic...")
    last_known_q = processed_df.iloc[-1]
    current_arr = last_known_q['cARR']
    future_quarters = ["FY26 Q1", "FY26 Q2", "FY26 Q3", "FY26 Q4"]
    forecast = []

    print(f"🔍 Starting ARR: ${current_arr:,.0f}")
    
    # Use the historical data to create proper YoY baselines
    # We have 5 quarters of data (4 historical + 1 current)
    historical_arrs = processed_df['cARR'].tolist()
    
    for i in range(4):
        predicted_yoy_growth = predicted_growth_rates[i]
        
        # For YoY growth, we need the ARR from the same quarter last year
        # Since we're predicting Q1-Q4 of 2026, we need Q1-Q4 of 2025 as baselines
        # We can use the historical data we generated as a proxy
        
        if i < len(historical_arrs):
            # Use historical data as baseline for YoY calculation
            baseline_arr = historical_arrs[i]  # Same quarter from historical data
            predicted_arr = baseline_arr * (1 + predicted_yoy_growth/100)
            
            # Calculate quarterly growth from previous quarter
            if i == 0:
                prev_arr = current_arr
            else:
                prev_arr = forecast[i-1]['Predicted ARR ($)']
            
            quarterly_growth_pct = ((predicted_arr - prev_arr) / prev_arr) * 100
            
            print(f"🔍 Q{i+1} Debug: YoY Growth = {predicted_yoy_growth:.1f}% | Baseline = ${baseline_arr:,.0f} | Predicted ARR = ${predicted_arr:,.0f} | QoQ Growth = {quarterly_growth_pct:.1f}%")
        else:
            # Fallback: use current ARR as baseline
            predicted_arr = current_arr * (1 + predicted_yoy_growth/100)
            quarterly_growth_pct = predicted_yoy_growth / 4  # Rough approximation
            print(f"🔍 Q{i+1} Debug: YoY Growth = {predicted_yoy_growth:.1f}% | Using current ARR as baseline | Predicted ARR = ${predicted_arr:,.0f}")
        
        # Calculate uncertainty bounds (±10%)
        uncertainty_factor = 0.10
        lower_bound = predicted_arr * (1 - uncertainty_factor)
        upper_bound = predicted_arr * (1 + uncertainty_factor)
        
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted ARR ($)": predicted_arr,
            "Lower Bound ($)": lower_bound,
            "Upper Bound ($)": upper_bound,
            "Quarterly Growth (%)": quarterly_growth_pct,
            "YoY Growth (%)": predicted_yoy_growth
        })
        
        # Update current ARR for next iteration
        current_arr = predicted_arr

    return pd.DataFrame(forecast)

def _calculate_fallback_growth(current_arr, yoy_growth):
    """
    Calculate realistic growth rates for extreme growth companies using a fallback approach.
    This is used when the model can't handle extreme growth rates (>100% YoY).
    """
    # Convert YoY growth to quarterly growth
    quarterly_growth = ((1 + yoy_growth/100) ** (1/4) - 1) * 100
    
    # For extreme growth companies, we expect growth to moderate over time
    # Start with high growth and gradually reduce it
    growth_rates = []
    
    for i in range(4):
        # Gradually reduce growth rate each quarter (growth companies typically slow down)
        # Start at 80% of the quarterly rate, then reduce by 10% each quarter
        quarter_growth = quarterly_growth * (0.8 - i * 0.1)
        quarter_growth = max(quarter_growth, 5.0)  # Minimum 5% growth
        growth_rates.append(quarter_growth)
    
    print(f"🔧 Fallback calculation:")
    print(f"  YoY growth: {yoy_growth:.1f}%")
    print(f"  Quarterly equivalent: {quarterly_growth:.1f}%")
    print(f"  Predicted quarterly rates: {[f'{g:.1f}%' for g in growth_rates]}")
    
    return growth_rates

def create_sample_company_data():
    """
    Creates sample data for a new company the model has never seen.
    This includes only the essential columns that are most likely to be available.
    """
    new_company_data = pd.DataFrame({
        "id_company": ["QuantumLeap Tech"] * 8,
        "Financial Quarter": [
            "FY24 Q1", "FY24 Q2", "FY24 Q3", "FY24 Q4",
            "FY25 Q1", "FY25 Q2", "FY25 Q3", "FY25 Q4"
        ],
        # Core Financials (cARR is the most important)
        "cARR": [
            10_000_000, 11_500_000, 13_200_000, 15_000_000,
            17_000_000, 19_500_000, 22_000_000, 25_000_000
        ],
        "ARR YoY Growth (in %)": [
            0.15, 0.15, 0.14, 0.13,  # More realistic historical values for FY24
            0.70, 0.69, 0.67, 0.66   # Current FY25 values (these are the targets)
        ],
        "Revenue YoY Growth (in %)": [
            0.12, 0.13, 0.12, 0.11,  # More realistic historical values
            0.68, 0.67, 0.65, 0.64
        ],
        "Gross Margin (in %)": [80, 81, 82, 81, 81, 80, 82, 81],
        "Sales & Marketing": [
            1_000_000, 1_100_000, 1_200_000, 1_300_000,
            1_500_000, 1_600_000, 1_800_000, 2_000_000
        ],
        "Cash Burn (OCF & ICF)": [
            -2_000_000, -1_800_000, -1_500_000, -1_200_000,
            -1_000_000, -800_000, -600_000, -500_000
        ],
        # Other operational metrics
        "Headcount (HC)": [80, 85, 92, 100, 110, 120, 135, 150],
        "Customers (EoP)": [200, 220, 245, 270, 300, 330, 365, 400],
        "Expansion & Upsell": [
            600_000, 650_000, 700_000, 750_000,
            800_000, 850_000, 900_000, 1_000_000
        ],
        "Churn & Reduction": [
            -100_000, -150_000, -100_000, -50_000,
            -100_000, -50_000, -100_000, -50_000
        ],
    })
    
    return new_company_data

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    
    # Load the trained model
    model_data = load_trained_model()
    
    if model_data is not None:
        # Create sample data for prediction
        new_company_data = create_sample_company_data()
        
        # Generate the forecast
        future_forecast_df = predict_future_arr(model_data, new_company_data)

        # Display the final forecast
        print("\n--- 🔮 FORECAST FOR QuantumLeap Tech ---")
        print(future_forecast_df.to_string(index=False))
        
        # Display model performance info
        print(f"\n--- 📊 MODEL PERFORMANCE SUMMARY ---")
        print(f"Overall R² Score: {model_data['overall_r2']:.4f}")
        print("\nIndividual Target Performance:")
        for target, metrics in model_data['performance_results'].items():
            print(f"  {target}: MAE = {metrics['MAE']:.4f}, R² = {metrics['R2']:.4f}")
    else:
        print("❌ Cannot proceed without a trained model. Please run the training script first.") 