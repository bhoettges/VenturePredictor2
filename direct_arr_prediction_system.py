#!/usr/bin/env python3
"""
Direct ARR Prediction System with Adaptive Defaults
==================================================

This system implements a sophisticated approach to predict direct ARR values
for the next 4 quarters using intelligent feature engineering and adaptive
defaults based on company characteristics.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings('ignore')

class DirectARRPredictionSystem:
    """Advanced ARR prediction system with adaptive defaults and smart feature engineering."""
    
    def __init__(self, model_path='lightgbm_financial_model.pkl'):
        """Initialize the prediction system."""
        self.model_path = model_path
        self.model_data = None
        self.training_data = None
        self.load_trained_model()
        self.load_training_data()
    
    def load_trained_model(self):
        """Load the trained model and its components."""
        print("Loading trained model...")
        try:
            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_training_data(self):
        """Load training data for adaptive defaults."""
        print("Loading training data for adaptive defaults...")
        try:
            self.training_data = pd.read_csv('202402_Copy.csv')
            print("✅ Training data loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load training data: {e}")
            self.training_data = None
    
    def load_and_clean_data(self, company_df):
        """Load and clean company data with proper preprocessing."""
        print("Step 1: Loading and preprocessing company data...")
        
        df_clean = company_df.copy()
        
        # --- Time Index Creation (must match training exactly) ---
        # Handle both "FY23 Q1" and "Q1 2023" formats
        def parse_quarter(quarter_str):
            try:
                # Try FY format first
                if 'FY' in quarter_str:
                    year_match = pd.Series([quarter_str]).str.extract(r'FY(\d{2,4})')[0]
                    if not year_match.isna().iloc[0]:
                        year = int(year_match.iloc[0])
                        year = year + 2000 if year < 100 else year
                        quarter_match = pd.Series([quarter_str]).str.extract(r'(Q[1-4])')[0]
                        quarter = quarter_match.map({'Q1':1,'Q2':2,'Q3':3,'Q4':4}).iloc[0]
                        return year * 4 + quarter
                
                # Try "Q1 2023" format
                if 'Q' in quarter_str and any(str(year) in quarter_str for year in range(2020, 2030)):
                    year_match = pd.Series([quarter_str]).str.extract(r'(\d{4})')[0]
                    if not year_match.isna().iloc[0]:
                        year = int(year_match.iloc[0])
                        quarter_match = pd.Series([quarter_str]).str.extract(r'(Q[1-4])')[0]
                        quarter = quarter_match.map({'Q1':1,'Q2':2,'Q3':3,'Q4':4}).iloc[0]
                        return year * 4 + quarter
                
                # Default fallback
                return 2023 * 4 + 1
            except:
                return 2023 * 4 + 1
        
        df_clean['time_idx'] = df_clean['Financial Quarter'].apply(parse_quarter)
        
        # Extract year and quarter for other uses
        df_clean['Year'] = df_clean['time_idx'] // 4
        df_clean['Quarter Num'] = df_clean['time_idx'] % 4
        df_clean = df_clean.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
        
        # --- Data Type Coercion ---
        potential_numeric_cols = df_clean.columns.drop(['Financial Quarter', 'id_company'])
        for col in potential_numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # --- Nuanced Imputation Strategy (must match training exactly) ---
        # Create 'Net New ARR' from cARR difference before filling other flow variables
        df_clean['Net New ARR'] = df_clean.groupby('id_company')['cARR'].transform(lambda x: x.diff())
        
        # Add missing columns with reasonable defaults based on available data
        if 'Sales & Marketing' not in df_clean.columns:
            df_clean['Sales & Marketing'] = df_clean['Net New ARR'] * 0.8  # Assume 80% efficiency
        if 'Cash Burn (OCF & ICF)' not in df_clean.columns:
            df_clean['Cash Burn (OCF & ICF)'] = -df_clean['cARR'] * 0.3  # Assume 30% burn rate
        if 'Revenue YoY Growth (in %)' not in df_clean.columns:
            df_clean['Revenue YoY Growth (in %)'] = df_clean.groupby('id_company')['cARR'].pct_change(4) * 100
        if 'Customers (EoP)' not in df_clean.columns:
            df_clean['Customers (EoP)'] = df_clean['Headcount (HC)'] * 0.1  # Assume 10% of headcount are customers
        if 'Expansion & Upsell' not in df_clean.columns:
            df_clean['Expansion & Upsell'] = df_clean['Net New ARR'] * 0.2  # Assume 20% expansion
        if 'Churn & Reduction' not in df_clean.columns:
            df_clean['Churn & Reduction'] = -df_clean['Net New ARR'] * 0.1  # Assume 10% churn
        
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
        if 'Gross Margin (in %)' in df_clean.columns:
            df_clean['Gross Margin (in %)'] = df_clean.groupby('id_company')['Gross Margin (in %)'].transform(lambda x: x.fillna(x.median()))
            df_clean['Gross Margin (in %)'].fillna(df_clean['Gross Margin (in %)'].median(), inplace=True)
        
        print("✅ Data cleaning complete")
        return df_clean
    
    def engineer_features(self, df):
        """Engineer a rich set of temporal and domain-specific SaaS features with bias correction."""
        print("Step 2: Engineering features with bias correction...")
        df_feat = df.copy()
        df_feat = df_feat.sort_values(by=['id_company', 'time_idx'])  # Ensure order
        
        # --- Temporal Features (Lags & Rolling Windows) ---
        metrics_to_process = [
            'cARR', 'Net New ARR', 'Cash Burn (OCF & ICF)', 'Gross Margin (in %)',
            'Sales & Marketing', 'Headcount (HC)', 'Revenue YoY Growth (in %)'
        ]
        lags = [1, 2, 4]  # 1 quarter, 2 quarters, 1 year
        
        for col in metrics_to_process:
            if col not in df_feat.columns:
                continue
            for lag in lags:
                df_feat[f'{col}_lag_{lag}'] = df_feat.groupby('id_company')[col].shift(lag)
        
        # 4-Quarter Rolling Stats (shifted to prevent data leakage from current period)
        for col in metrics_to_process:
            if col not in df_feat.columns:
                continue
            df_feat[f'{col}_roll_mean_4q'] = df_feat.groupby('id_company')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
            df_feat[f'{col}_roll_std_4q'] = df_feat.groupby('id_company')[col].transform(
                lambda x: x.rolling(window=4, min_periods=1).std().shift(1))
        
        # --- Advanced SaaS Efficiency Metrics ---
        # Magic Number: New ARR generated for every $1 of S&M spend from the prior quarter.
        sm_spend_lag1 = df_feat.groupby('id_company')['Sales & Marketing'].shift(1)
        df_feat['Magic_Number'] = df_feat['Net New ARR'] / (sm_spend_lag1 + 1e-8)
        
        # Burn Multiple: How much cash is burned to generate $1 of new ARR.
        df_feat['Burn_Multiple'] = np.abs(df_feat['Cash Burn (OCF & ICF)']) / (df_feat['Net New ARR'] + 1e-8)
        
        # --- Other Growth & Ratio Features ---
        df_feat['HC_qoq_growth'] = df_feat.groupby('id_company')['Headcount (HC)'].pct_change(1)
        df_feat['ARR_per_Headcount'] = df_feat['cARR'] / (df_feat['Headcount (HC)'] + 1)
        
        # --- Q1 Bias Correction Features ---
        # Create quarter indicators
        df_feat['is_q1'] = (df_feat['Quarter Num'] == 1).astype(int)
        df_feat['is_q2'] = (df_feat['Quarter Num'] == 2).astype(int)
        df_feat['is_q3'] = (df_feat['Quarter Num'] == 3).astype(int)
        df_feat['is_q4'] = (df_feat['Quarter Num'] == 4).astype(int)
        
        # Create seasonality features
        df_feat['quarter_sin'] = np.sin(2 * np.pi * df_feat['Quarter Num'] / 4)
        df_feat['quarter_cos'] = np.cos(2 * np.pi * df_feat['Quarter Num'] / 4)
        
        # --- Growth Rate Normalization Features ---
        # Calculate realistic growth rates with outlier capping
        df_feat['yoy_growth_raw'] = df_feat.groupby('id_company')['cARR'].pct_change(4)
        
        # Cap extreme growth rates to realistic ranges
        df_feat['yoy_growth_capped'] = df_feat['yoy_growth_raw'].clip(-0.5, 1.0)  # -50% to +100%
        
        # Create growth momentum features
        df_feat['growth_momentum'] = df_feat.groupby('id_company')['yoy_growth_capped'].diff(1)
        df_feat['growth_acceleration'] = df_feat.groupby('id_company')['growth_momentum'].diff(1)
        
        # --- Company Size and Maturity Features ---
        # ARR size buckets
        df_feat['arr_size_small'] = (df_feat['cARR'] < 1e6).astype(int)
        df_feat['arr_size_medium'] = ((df_feat['cARR'] >= 1e6) & (df_feat['cARR'] < 10e6)).astype(int)
        df_feat['arr_size_large'] = (df_feat['cARR'] >= 10e6).astype(int)
        
        # Growth stage features
        df_feat['is_high_growth'] = (df_feat['yoy_growth_capped'] > 0.3).astype(int)  # >30% growth
        df_feat['is_stable_growth'] = ((df_feat['yoy_growth_capped'] >= 0.1) & (df_feat['yoy_growth_capped'] <= 0.3)).astype(int)
        df_feat['is_low_growth'] = (df_feat['yoy_growth_capped'] < 0.1).astype(int)
        
        # --- Efficiency Trend Features ---
        # Magic number trends
        df_feat['magic_number_trend'] = df_feat.groupby('id_company')['Magic_Number'].diff(1)
        df_feat['magic_number_improving'] = (df_feat['magic_number_trend'] > 0).astype(int)
        
        # Burn multiple trends
        df_feat['burn_multiple_trend'] = df_feat.groupby('id_company')['Burn_Multiple'].diff(1)
        df_feat['burn_multiple_improving'] = (df_feat['burn_multiple_trend'] < 0).astype(int)
        
        # --- Clean up generated features ---
        numeric_cols = df_feat.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            # Replace infinities from divisions with NaN
            df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan)
            # More conservative winsorization for growth rates
            if 'growth' in col.lower() or 'yoy' in col.lower():
                p05 = df_feat[col].quantile(0.05)
                p95 = df_feat[col].quantile(0.95)
                df_feat[col] = df_feat[col].clip(p05, p95)
            else:
                # Standard winsorization for other features
                p01 = df_feat[col].quantile(0.01)
                p99 = df_feat[col].quantile(0.99)
                df_feat[col] = df_feat[col].clip(p01, p99)
        
        print("✅ Feature engineering with bias correction complete")
        return df_feat
    
    def _apply_realistic_constraints(self, growth_rates, current_arr):
        """Apply realistic constraints to growth rate predictions."""
        constrained_rates = growth_rates.copy()
        
        # Define realistic growth rate bounds based on company size and stage
        if current_arr < 1e6:  # Small company
            max_growth = 0.8  # 80% max growth
            min_growth = -0.3  # -30% min growth
        elif current_arr < 10e6:  # Medium company
            max_growth = 0.5  # 50% max growth
            min_growth = -0.2  # -20% min growth
        else:  # Large company
            max_growth = 0.3  # 30% max growth
            min_growth = -0.1  # -10% min growth
        
        # Apply constraints
        constrained_rates = np.clip(constrained_rates, min_growth, max_growth)
        
        # Apply Q1 bias correction (reduce Q1 predictions by 3 percentage points)
        if len(constrained_rates) > 0:
            # Assume first prediction is Q1
            constrained_rates[0] = constrained_rates[0] - 0.03
        
        # Apply growth rate decay (later quarters should have lower growth)
        for i in range(1, len(constrained_rates)):
            constrained_rates[i] = constrained_rates[i] * (0.95 ** i)  # 5% decay per quarter
        
        # Ensure monotonic decay for stability
        for i in range(1, len(constrained_rates)):
            if constrained_rates[i] > constrained_rates[i-1]:
                constrained_rates[i] = constrained_rates[i-1] * 0.9  # 10% reduction
        
        return constrained_rates
    
    def get_adaptive_defaults(self, prediction_input_row):
        """Get adaptive defaults based on user input characteristics and learned relationships."""
        print("Step 1.5: Smart feature imputation...")
        
        if self.training_data is None:
            print("📊 Using conservative fallback defaults...")
            return self._get_fallback_defaults(prediction_input_row)
        
        try:
            # Calculate key metrics from training data
            training_df = self.training_data.copy()
            
            # Calculate ARR YoY Growth if not present
            if 'ARR YoY Growth (in %)' not in training_df.columns:
                training_df['ARR YoY Growth (in %)'] = training_df.groupby('id_company')['cARR'].pct_change(4) * 100
            
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
                'Churn & Reduction': 'median'
            }).fillna(method='ffill')
            
            growth_metrics = training_df[training_df['growth_category'] == user_growth_category].agg({
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
                'Headcount (HC)': max(1, int(user_arr / 150000)),  # Conservative ARR per headcount
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
            print(f"  Sales & Marketing: ${adaptive_defaults['Sales & Marketing']:,.0f}")
            
            return adaptive_defaults
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load training data for adaptive defaults: {e}")
            print("📊 Using conservative fallback defaults...")
            return self._get_fallback_defaults(prediction_input_row)
    
    def _get_fallback_defaults(self, prediction_input_row):
        """Fallback to conservative defaults based on user's ARR."""
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
            
            # Time-based features
            'Quarter Num': prediction_input_row['Quarter Num'].iloc[0] if 'Quarter Num' in prediction_input_row.columns else 1,
            'time_idx': prediction_input_row['time_idx'].iloc[0] if 'time_idx' in prediction_input_row.columns else 1,
            
            # Categorical features (use 0 for unknown/neutral)
            'Currency': 0, 'id_currency': 0, 'Sector': 0, 'id_sector': 0,
            'Target Customer': 0, 'id_target_customer': 0, 'Country': 0, 'id_country': 0,
            'Deal Team': 0, 'id_deal_team': 0
        }
    
    def predict_future_arr(self, company_df):
        """Predict future ARR growth rates for a company using the trained model."""
        print("Step 1: Loading and preprocessing company data...")
        
        # Load the trained model
        model_pipeline = self.model_data['model_pipeline']
        feature_cols = self.model_data['feature_cols']
        
        # Process the company data directly
        print("Step 1.1: Processing company data...")
        
        # Apply the same cleaning logic as in load_and_clean_data
        df_clean = self.load_and_clean_data(company_df)
        
        # Process the company data (same as training)
        processed_df = self.engineer_features(df_clean)
        
        # Get the most recent quarter for prediction
        prediction_input_row = processed_df.iloc[-1:].copy()
        
        # Create feature matrix for prediction
        X_predict = pd.DataFrame(index=prediction_input_row.index)
        
        # Smart imputation based on training data patterns
        adaptive_defaults = self.get_adaptive_defaults(prediction_input_row)
        
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
        
        # Get the last year's ARR for YoY growth calculation
        last_year_arr = prediction_input_row['cARR'].iloc[0]
        
        print("Step 2: Making predictions with the trained model...")
        predicted_growth_rates = model_pipeline.predict(X_predict)[0]
        
        print(f"🔍 Debug: Raw predictions from model = {predicted_growth_rates}")
        print(f"🔍 Debug: Predictions as percentages = {[f'{x*100:.1f}%' for x in predicted_growth_rates]}")
        
        # Apply realistic growth rate constraints
        print("Step 2.1: Applying realistic growth rate constraints...")
        predicted_growth_rates = self._apply_realistic_constraints(predicted_growth_rates, last_year_arr)
        
        print(f"🔍 Debug: Constrained predictions = {predicted_growth_rates}")
        print(f"🔍 Debug: Constrained percentages = {[f'{x*100:.1f}%' for x in predicted_growth_rates]}")
        print(f"🔍 Debug: Current ARR = ${last_year_arr:,.0f}")
        
        # --- Translate growth rates into absolute ARR values ---
        print("Step 3: Calculating future absolute ARR...")
        
        # Define future quarters
        current_quarter = int(prediction_input_row['Quarter Num'].iloc[0])
        current_year = int(prediction_input_row['Year'].iloc[0])
        
        future_quarters = []
        for i in range(4):
            future_q = current_quarter + i + 1
            future_year = current_year
            if future_q > 4:
                future_q -= 4
                future_year += 1
            future_quarters.append(f"Q{future_q} {future_year}")
        
        # Calculate future ARR values
        forecast = []
        predicted_arr = last_year_arr
        
        for i, predicted_growth in enumerate(predicted_growth_rates):
            predicted_arr = last_year_arr * (1 + predicted_growth)
            forecast.append({
                "Future Quarter": future_quarters[i],
                "Predicted YoY Growth (%)": predicted_growth * 100,  # Keep as numeric for calculations
                "Predicted Absolute cARR ($)": predicted_arr
            })
        
        return pd.DataFrame(forecast)

def main():
    """Main function to test the direct ARR prediction system."""
    print("Direct ARR Prediction System with Adaptive Defaults")
    print("=" * 60)
    
    # Initialize the system
    predictor = DirectARRPredictionSystem()
    
    # Load test company data
    print("\nLoading test company data...")
    company_df = pd.read_csv('test_company_2024.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_2024'
    
    print("Company Data:")
    print(company_df[['Financial Quarter', 'cARR', 'Headcount (HC)', 'Gross Margin (in %)']])
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict_future_arr(company_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    for _, row in predictions.iterrows():
        print(f"{row['Future Quarter']}:")
        print(f"  Growth Rate: {row['Predicted YoY Growth (%)']:.1f}%")
        print(f"  ARR: ${row['Predicted Absolute cARR ($)']:,.0f}")
        print()
    
    print("✅ Direct ARR prediction complete!")

if __name__ == "__main__":
    main()
