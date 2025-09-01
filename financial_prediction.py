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
    
    # Smart imputation based on training data patterns
    print("Step 1.5: Smart feature imputation...")
    
    def _get_adaptive_defaults(prediction_input_row):
        """
        Get adaptive defaults based on user input characteristics and learned relationships from training data.
        """
        try:
            # Load training data to learn relationships
            training_df = pd.read_csv('202402_Copy.csv')
            
            # Calculate key metrics from training data
            training_df['ARR_size_category'] = pd.cut(training_df['cARR'], 
                                                   bins=[0, 1e6, 10e6, 100e6, np.inf],
                                                   labels=['Small', 'Medium', 'Large', 'Enterprise'])
            
            training_df['growth_category'] = pd.cut(training_df['ARR YoY Growth (in %)'], 
                                                 bins=[-np.inf, 0, 20, 50, 100, np.inf],
                                                 labels=['Declining', 'Slow', 'Moderate', 'Fast', 'Hyper'])
            
            # Get user's ARR and growth characteristics
            user_arr = prediction_input_row['cARR'].iloc[0] if 'cARR' in prediction_input_row.columns else 1000000
            user_growth = prediction_input_row['ARR YoY Growth (in %)'].iloc[0] if 'ARR YoY Growth (in %)' in prediction_input_row.columns else 20
            
            # Determine user's size and growth categories
            if user_arr < 1e6:
                user_size_category = 'Small'
            elif user_arr < 10e6:
                user_size_category = 'Medium'
            elif user_arr < 100e6:
                user_size_category = 'Large'
            else:
                user_size_category = 'Enterprise'
                
            if user_growth < 0:
                user_growth_category = 'Declining'
            elif user_growth < 20:
                user_growth_category = 'Slow'
            elif user_growth < 50:
                user_growth_category = 'Moderate'
            elif user_growth < 100:
                user_growth_category = 'Fast'
            else:
                user_growth_category = 'Hyper'
            
            # Calculate adaptive metrics based on size and growth
            size_metrics = training_df[training_df['ARR_size_category'] == user_size_category].agg({
                'Gross Margin (in %)': 'median',
                'Net Profit/Loss Margin (in %)': 'median',
                'Headcount (HC)': 'median',
                'Customers (EoP)': 'median',
                'Sales & Marketing': 'median',
                'R&D': 'median',
                'G&A': 'median',
                'Cash Burn (OCF & ICF)': 'median',
                'Expansion & Upsell': 'median',
                'Churn & Reduction': 'median',
                'Magic Number': 'median',
                'Burn Multiple': 'median',
                'ARR per Headcount': 'median'
            }).fillna(method='ffill')
            
            growth_metrics = training_df[training_df['growth_category'] == user_growth_category].agg({
                'Magic Number': 'median',
                'Burn Multiple': 'median',
                'ARR per Headcount': 'median',
                'Sales & Marketing': 'median'
            }).fillna(method='ffill')
            
            # Get most common categorical values from training data
            categorical_defaults = {
                'Currency': training_df['Currency'].mode().iloc[0] if len(training_df['Currency'].mode()) > 0 else 0,
                'id_currency': training_df['id_currency'].mode().iloc[0] if len(training_df['id_currency'].mode()) > 0 else 0,
                'Sector': training_df['Sector'].mode().iloc[0] if len(training_df['Sector'].mode()) > 0 else 0,
                'id_sector': training_df['id_sector'].mode().iloc[0] if len(training_df['id_sector'].mode()) > 0 else 0,
                'Target Customer': training_df['Target Customer'].mode().iloc[0] if len(training_df['Target Customer'].mode()) > 0 else 0,
                'id_target_customer': training_df['id_target_customer'].mode().iloc[0] if len(training_df['id_target_customer'].mode()) > 0 else 0,
                'Country': training_df['Country'].mode().iloc[0] if len(training_df['Country'].mode()) > 0 else 0,
                'id_country': training_df['id_country'].mode().iloc[0] if len(training_df['id_country'].mode()) > 0 else 0,
                'Deal Team': training_df['Deal Team'].mode().iloc[0] if len(training_df['Deal Team'].mode()) > 0 else 0,
                'id_deal_team': training_df['id_deal_team'].mode().iloc[0] if len(training_df['id_deal_team'].mode()) > 0 else 0
            }
            
            print(f"🔧 Categorical defaults from training data:")
            print(f"  Sector: {categorical_defaults['Sector']} (most common)")
            print(f"  Country: {categorical_defaults['Country']} (most common)")
            print(f"  Currency: {categorical_defaults['Currency']} (most common)")
            
            # Combine size and growth metrics with weighted averages
            adaptive_defaults = {
                # Core financial metrics (adapted to user's size and growth)
                'cARR': user_arr,
                'ARR YoY Growth (in %)': user_growth,
                'Revenue YoY Growth (in %)': user_growth * 0.8,  # Revenue typically grows slower than ARR
                'Gross Margin (in %)': size_metrics.get('Gross Margin (in %)', 75),
                'Net Profit/Loss Margin (in %)': size_metrics.get('Net Profit/Loss Margin (in %)', -10),
                
                # Headcount and operational metrics (scaled by ARR)
                'Headcount (HC)': max(1, int(user_arr / size_metrics.get('ARR per Headcount', 100000))),
                'Customers (EoP)': size_metrics.get('Customers (EoP)', 200),
                'Sales & Marketing': (size_metrics.get('Sales & Marketing', 2000000) + growth_metrics.get('Sales & Marketing', 2000000)) / 2,
                'R&D': size_metrics.get('R&D', 1500000),
                'G&A': size_metrics.get('G&A', 1000000),
                
                # Cash flow metrics (adapted to growth rate)
                'Cash Burn (OCF & ICF)': size_metrics.get('Cash Burn (OCF & ICF)', -1000000),
                'Net Cash Flow': size_metrics.get('Cash Burn (OCF & ICF)', -1000000) * 0.8,
                
                # Customer metrics (scaled by ARR)
                'Expansion & Upsell': size_metrics.get('Expansion & Upsell', 500000) * (user_arr / 1000000),
                'Churn & Reduction': size_metrics.get('Churn & Reduction', -100000) * (user_arr / 1000000),
                
                # Efficiency metrics (combined size and growth)
                'Magic Number': (size_metrics.get('Magic Number', 0.8) + growth_metrics.get('Magic Number', 0.8)) / 2,
                'Burn Multiple': (size_metrics.get('Burn Multiple', 1.2) + growth_metrics.get('Burn Multiple', 1.2)) / 2,
                'ARR per Headcount': (size_metrics.get('ARR per Headcount', 100000) + growth_metrics.get('ARR per Headcount', 100000)) / 2,
                
                # Time-based features
                'Quarter Num': prediction_input_row['Quarter Num'].iloc[0] if 'Quarter Num' in prediction_input_row.columns else 1,
                'time_idx': prediction_input_row['time_idx'].iloc[0] if 'time_idx' in prediction_input_row.columns else 1,
                
                # Categorical features (use most common values from training data)
                **categorical_defaults
            }
            
            print(f"🔧 Adaptive defaults for {user_size_category} company with {user_growth_category} growth:")
            print(f"  ARR: ${user_arr:,.0f}")
            print(f"  Growth: {user_growth:.1f}%")
            print(f"  Headcount: {adaptive_defaults['Headcount (HC)']}")
            print(f"  Magic Number: {adaptive_defaults['Magic Number']:.2f}")
            print(f"  Burn Multiple: {adaptive_defaults['Burn Multiple']:.2f}")
            
            return adaptive_defaults
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load training data for adaptive defaults: {e}")
            print("📊 Using conservative fallback defaults...")
            
            # Fallback to conservative defaults based on user's ARR
            user_arr = prediction_input_row['cARR'].iloc[0] if 'cARR' in prediction_input_row.columns else 1000000
            
            return {
                # Core financial metrics
                'cARR': user_arr,
                'ARR YoY Growth (in %)': 20,  # Conservative 20% growth
                'Revenue YoY Growth (in %)': 16,  # 80% of ARR growth
                'Gross Margin (in %)': 75,
                'Net Profit/Loss Margin (in %)': -15,
                
                # Headcount and operational metrics (scaled by ARR)
                'Headcount (HC)': max(1, int(user_arr / 150000)),  # Conservative ARR per headcount
                'Customers (EoP)': max(1, int(user_arr / 5000)),  # Conservative customer count
                'Sales & Marketing': user_arr * 0.4,  # 40% of ARR
                'R&D': user_arr * 0.25,  # 25% of ARR
                'G&A': user_arr * 0.15,  # 15% of ARR
                
                # Cash flow metrics
                'Cash Burn (OCF & ICF)': -user_arr * 0.3,  # 30% burn rate
                'Net Cash Flow': -user_arr * 0.25,  # 25% net burn
                
                # Customer metrics
                'Expansion & Upsell': user_arr * 0.1,  # 10% expansion
                'Churn & Reduction': -user_arr * 0.05,  # 5% churn
                
                # Efficiency metrics
                'Magic Number': 0.6,  # Conservative magic number
                'Burn Multiple': 1.5,  # Conservative burn multiple
                'ARR per Headcount': 150000,  # Conservative ARR per headcount
                
                # Time-based features
                'Quarter Num': prediction_input_row['Quarter Num'].iloc[0] if 'Quarter Num' in prediction_input_row.columns else 1,
                'time_idx': prediction_input_row['time_idx'].iloc[0] if 'time_idx' in prediction_input_row.columns else 1,
                
                # Categorical features (use 0 for unknown/neutral)
                'Currency': 0,
                'id_currency': 0,
                'Sector': 0,
                'id_sector': 0,
                'Target Customer': 0,
                'id_target_customer': 0,
                'Country': 0,
                'id_country': 0,
                'Deal Team': 0,
                'id_deal_team': 0
            }

    adaptive_defaults = _get_adaptive_defaults(prediction_input_row)
    
    for col in feature_cols:
        if col in prediction_input_row.columns:
            X_predict[col] = prediction_input_row[col]
        elif col in adaptive_defaults:
            X_predict[col] = adaptive_defaults[col]
            print(f"📊 Using adaptive default for '{col}': {adaptive_defaults[col]}")
        else:
            # For unknown features, use 0 but log it
            X_predict[col] = 0
            print(f"⚠️  Warning: Unknown feature '{col}', using 0")

    # Fill any remaining NaNs with 0
    X_predict = X_predict.fillna(0)
    
    print(f"🔍 Debug: Input features shape: {X_predict.shape}")
    print(f"🔍 Debug: Sample input values: {X_predict.iloc[0, :5].to_dict()}")

    print("Step 2: Making predictions with the trained model...")
    predicted_growth_rates = model_pipeline.predict(X_predict)[0]
    
    print(f"🔍 Debug: Raw predictions from model = {predicted_growth_rates}")
    
    # The model should output growth rates in the same scale it was trained on
    # Let's see what the actual values are without arbitrary scaling
    print(f"🔍 Debug: Model predictions (no scaling) = {predicted_growth_rates}")

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
            "Predicted YoY Growth (%)": predicted_growth * 100,  # Keep as numeric for calculations
            "Predicted Absolute cARR (€)": predicted_arr
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