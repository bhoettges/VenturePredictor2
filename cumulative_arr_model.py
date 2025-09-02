#!/usr/bin/env python3
"""
Cumulative ARR Prediction Model
==============================

Build a model that predicts cumulative ARR for the next year instead of volatile QoQ/YoY growth rates.
This approach should be more stable and business-friendly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class CumulativeARRModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_cols = None
        self.target_cols = None
        
    def load_and_prepare_data(self):
        """Load and prepare the training data for cumulative ARR prediction."""
        print("Loading and preparing data for cumulative ARR prediction...")
        
        # Load the data
        df = pd.read_csv('202402_Copy.csv')
        print(f"Loaded {len(df)} records")
        
        # Create time index
        # Extract year from Financial Quarter (e.g., Q1-FY19 -> 2019)
        df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2})')
        df['Year'] = df['Year'].astype(float)
        df['Year'] = df['Year'] + 2000  # Convert 19 -> 2019, 20 -> 2020, etc.
        df['Year'] = df['Year'].fillna(2020).astype(int)  # Default to 2020 for missing values
        
        # Extract quarter number
        df['Quarter Num'] = df['Financial Quarter'].str.extract(r'Q(\d)').astype(int)
        df['Quarter Num'] = df['Quarter Num'].fillna(1)  # Default to Q1 for missing values
        
        df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
        
        # Sort by company and time
        df = df.sort_values(['id_company', 'time_idx'])
        
        # Calculate cumulative ARR for each company
        df['Cumulative_ARR'] = df.groupby('id_company')['cARR'].cumsum()
        
        # Create targets: cumulative ARR for next 4 quarters
        df['Future_Cumulative_ARR_Q1'] = df.groupby('id_company')['Cumulative_ARR'].shift(-1)
        df['Future_Cumulative_ARR_Q2'] = df.groupby('id_company')['Cumulative_ARR'].shift(-2)
        df['Future_Cumulative_ARR_Q3'] = df.groupby('id_company')['Cumulative_ARR'].shift(-3)
        df['Future_Cumulative_ARR_Q4'] = df.groupby('id_company')['Cumulative_ARR'].shift(-4)
        
        # Calculate cumulative ARR growth rates (more stable than QoQ/YoY)
        df['Cumulative_ARR_Growth_Q1'] = (df['Future_Cumulative_ARR_Q1'] - df['Cumulative_ARR']) / df['Cumulative_ARR']
        df['Cumulative_ARR_Growth_Q2'] = (df['Future_Cumulative_ARR_Q2'] - df['Cumulative_ARR']) / df['Cumulative_ARR']
        df['Cumulative_ARR_Growth_Q3'] = (df['Future_Cumulative_ARR_Q3'] - df['Cumulative_ARR']) / df['Cumulative_ARR']
        df['Cumulative_ARR_Growth_Q4'] = (df['Future_Cumulative_ARR_Q4'] - df['Cumulative_ARR']) / df['Cumulative_ARR']
        
        # Remove rows with missing targets
        df = df.dropna(subset=['Future_Cumulative_ARR_Q1', 'Future_Cumulative_ARR_Q2', 
                              'Future_Cumulative_ARR_Q3', 'Future_Cumulative_ARR_Q4'])
        
        print(f"After removing missing targets: {len(df)} records")
        
        # Define feature columns (matching actual dataset)
        feature_cols = [
            'cARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Net Income',
            'Customers (EoP)', 'Sales & Marketing', 'Research & Development', 'General & Administrative', 
            'Cash Burn (OCF & ICF)', 'Expansion & Upsell', 'Churn & Reduction', 'Cumulative_ARR'
        ]
        
        # Add lag features
        for col in ['cARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Net Income']:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('id_company')[col].shift(1)
                df[f'{col}_lag2'] = df.groupby('id_company')[col].shift(2)
                df[f'{col}_lag3'] = df.groupby('id_company')[col].shift(3)
                feature_cols.extend([f'{col}_lag1', f'{col}_lag2', f'{col}_lag3'])
        
        # Add rolling window features
        for col in ['cARR', 'Headcount (HC)']:
            if col in df.columns:
                df[f'{col}_rolling3'] = df.groupby('id_company')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_rolling4'] = df.groupby('id_company')[col].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
                feature_cols.extend([f'{col}_rolling3', f'{col}_rolling4'])
        
        # Add efficiency ratios
        if 'Sales & Marketing' in df.columns and 'cARR' in df.columns:
            df['Magic_Number'] = df['Sales & Marketing'] / df['cARR']
            feature_cols.append('Magic_Number')
        
        if 'Cash Burn (OCF & ICF)' in df.columns and 'cARR' in df.columns:
            df['Burn_Multiple'] = abs(df['Cash Burn (OCF & ICF)']) / df['cARR']
            feature_cols.append('Burn_Multiple')
        
        # Add temporal features
        df['quarter_sin'] = np.sin(2 * np.pi * df['Quarter Num'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['Quarter Num'] / 4)
        df['is_q1'] = (df['Quarter Num'] == 1).astype(int)
        feature_cols.extend(['quarter_sin', 'quarter_cos', 'is_q1'])
        
        # Add company size features
        df['arr_size_small'] = (df['cARR'] < 1000000).astype(int)
        df['arr_size_medium'] = ((df['cARR'] >= 1000000) & (df['cARR'] < 10000000)).astype(int)
        df['arr_size_large'] = (df['cARR'] >= 10000000).astype(int)
        feature_cols.extend(['arr_size_small', 'arr_size_medium', 'arr_size_large'])
        
        # Add cumulative ARR growth momentum
        df['cumulative_arr_growth_momentum'] = df.groupby('id_company')['Cumulative_ARR'].pct_change(1)
        df['cumulative_arr_growth_momentum'] = df.groupby('id_company')['cumulative_arr_growth_momentum'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        feature_cols.append('cumulative_arr_growth_momentum')
        
        # Remove rows with missing features
        df = df.dropna(subset=feature_cols)
        
        print(f"After removing missing features: {len(df)} records")
        
        # Define target columns (NO CAPS - let the model learn real patterns)
        target_cols = ['Cumulative_ARR_Growth_Q1', 'Cumulative_ARR_Growth_Q2', 
                      'Cumulative_ARR_Growth_Q3', 'Cumulative_ARR_Growth_Q4']
        
        # Remove any infinite values but keep extreme finite values
        for col in target_cols:
            if col in df.columns:
                # Only remove infinite values, keep all finite values (even extreme ones)
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df = df.dropna(subset=[col])
        
        # Store the data
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        
        print(f"Final dataset: {len(df)} records, {len(feature_cols)} features")
        print(f"Target columns: {target_cols}")
        
        return df
    
    def train_model(self):
        """Train the cumulative ARR prediction model."""
        print("\nTraining cumulative ARR prediction model...")
        
        # Prepare features and targets
        X = self.df[self.feature_cols].fillna(0)
        y = self.df[self.target_cols].fillna(0)
        
        # Temporal split (use last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_regression, k=min(50, len(self.feature_cols)))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train.mean(axis=1))
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_selected.shape[1]} features")
        
        # Train separate models for each target
        self.models = {}
        results = {}
        
        for i, target_col in enumerate(self.target_cols):
            print(f"\nTraining model for {target_col}...")
            
            # LightGBM parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            # Train model
            train_data = lgb.Dataset(X_train_selected, label=y_train.iloc[:, i])
            self.models[target_col] = lgb.train(params, train_data, num_boost_round=100)
            
            # Make predictions
            y_pred = self.models[target_col].predict(X_test_selected)
            
            # Calculate metrics
            r2 = r2_score(y_test.iloc[:, i], y_pred)
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred)
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred))
            
            results[target_col] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            print(f"  R²: {r2:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
        
        # Overall performance
        avg_r2 = np.mean([results[col]['r2'] for col in self.target_cols])
        avg_mae = np.mean([results[col]['mae'] for col in self.target_cols])
        
        print(f"\nOverall Performance:")
        print(f"  Average R²: {avg_r2:.3f}")
        print(f"  Average MAE: {avg_mae:.3f}")
        
        return results
    
    def save_model(self, filename='cumulative_arr_model.pkl'):
        """Save the trained model."""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def predict_cumulative_arr(self, company_data):
        """Predict cumulative ARR growth for a company."""
        # Prepare features
        X = company_data[self.feature_cols].fillna(0)
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Make predictions
        predictions = {}
        for target_col in self.target_cols:
            pred = self.models[target_col].predict(X_selected)
            predictions[target_col] = pred[0]
        
        return predictions

def main():
    """Main function to train the cumulative ARR model."""
    print("CUMULATIVE ARR PREDICTION MODEL")
    print("=" * 50)
    
    # Initialize model
    model = CumulativeARRModel()
    
    # Load and prepare data
    df = model.load_and_prepare_data()
    
    # Train model
    results = model.train_model()
    
    # Save model
    model.save_model()
    
    print(f"\n✅ Cumulative ARR model training complete!")
    print(f"Model saved as 'cumulative_arr_model.pkl'")

if __name__ == "__main__":
    main()
