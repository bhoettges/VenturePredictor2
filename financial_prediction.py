# ==============================================================================
# FINANCIAL FORECASTING PREDICTION SCRIPT
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

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
    (This should be identical to the function used for training the model)
    """
    df_feat = df.copy()

    # --- Time Index Creation ---
    df_feat['Year'] = df_feat['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df_feat['Year'] = df_feat['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df_feat['Quarter Num'] = df_feat['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df_feat['time_idx'] = df_feat['Year'] * 4 + df_feat['Quarter Num']
    df_feat = df_feat.sort_values(by=['id_company', 'time_idx'])

    # --- Feature Creation ---
    df_feat['Net New ARR'] = df_feat.groupby('id_company')['cARR'].transform(lambda x: x.diff())
    metrics_to_process = [
        'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
        'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
    ]
    lags = [1, 2, 4]
    for col in metrics_to_process:
        if col not in df_feat.columns: continue
        for lag in lags:
            df_feat[f'{col}_lag_{lag}'] = df_feat.groupby('id_company')[col].shift(lag)
        df_feat[f'{col}_roll_mean_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
        df_feat[f'{col}_roll_std_4q'] = df_feat.groupby('id_company')[col].transform(lambda x: x.rolling(window=4, min_periods=1).std().shift(1))
    
    sm_spend_lag1 = df_feat.groupby('id_company')['Sales & Marketing'].shift(1)
    df_feat['Magic_Number'] = df_feat['Net New ARR'] / sm_spend_lag1
    df_feat['Burn_Multiple'] = np.abs(df_feat['Cash Burn (OCF & ICF)']) / df_feat['Net New ARR']
    df_feat['HC_qoq_growth'] = df_feat.groupby('id_company')['Headcount (HC)'].pct_change(1)
    df_feat['ARR_per_Headcount'] = df_feat['cARR'] / df_feat['Headcount (HC)']
    
    # Clean up infinities
    numeric_cols = df_feat.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
    
    return df_feat

def predict_future_arr(model_data, company_df):
    """
    Takes a trained model and a new company's historical data to forecast future ARR.

    Args:
        model_data: Dictionary containing the trained model and metadata.
        company_df: DataFrame with the new company's historical data.

    Returns:
        A pandas DataFrame with the forecast.
    """
    print("Step 1: Engineering features for the new company...")
    processed_df = engineer_features(company_df)

    # Get the model and feature columns
    model_pipeline = model_data['model_pipeline']
    feature_cols = model_data['feature_cols']

    # The last row contains the most recent data needed for prediction
    prediction_input_row = processed_df.tail(1)

    # Create a DataFrame with all required features, filling missing ones with 0
    X_predict = pd.DataFrame(index=prediction_input_row.index)
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        else:
            # Fill missing features with 0 (or you could use median values from training)
            X_predict[col] = 0
            print(f"⚠️  Warning: Feature '{col}' not found in new data, using 0")

    X_predict = X_predict.fillna(0) # Fill any remaining NaNs with 0 for prediction

    print("Step 2: Making predictions with the trained model...")
    predicted_growth_rates = model_pipeline.predict(X_predict)[0]

    # --- Translate growth rates into absolute ARR values ---
    print("Step 3: Calculating future absolute ARR...")
    last_known_q = processed_df.iloc[-1]
    future_quarters = ["FY26 Q1", "FY26 Q2", "FY26 Q3", "FY26 Q4"]
    forecast = []

    for i in range(4):
        # Get the corresponding quarter from the previous year for YoY calculation
        last_year_quarter_data = processed_df[processed_df['time_idx'] == last_known_q['time_idx'] - 3 + i]
        if len(last_year_quarter_data) > 0:
            last_year_arr = last_year_quarter_data['cARR'].iloc[0]
        else:
            # If we don't have the exact quarter, use the last known ARR
            last_year_arr = last_known_q['cARR']

        # Calculate predicted ARR
        predicted_growth = predicted_growth_rates[i]
        
        # Debug: Print the raw prediction and calculation
        print(f"🔍 Q{i+1} Debug: Raw prediction = {predicted_growth:.3f}, Last year ARR = {last_year_arr:,.0f}")
        
        predicted_arr = last_year_arr * (1 + predicted_growth)
        forecast.append({
            "Future Quarter": future_quarters[i],
            "Predicted YoY Growth (%)": f"{predicted_growth:.1%}",
            "Predicted Absolute cARR (€)": f"{predicted_arr:,.0f}"
        })

    return pd.DataFrame(forecast)

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