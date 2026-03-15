#!/usr/bin/env python3
"""
Intelligent Feature Completion System
====================================

This system intelligently infers missing features from user input by:
1. Analyzing company characteristics (size, growth, sector)
2. Using industry benchmarks and patterns
3. Learning from similar companies in the training data
4. Creating realistic synthetic features that match business logic
"""

import re
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def _sanitize_name(name: str) -> str:
    """Apply the same sanitization used during training."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

class IntelligentFeatureCompletionSystem:
    """Advanced feature completion system using company profiling and industry patterns."""
    
    def __init__(self, training_data_path='202402_Copy_Fixed.csv', model_path='lightgbm_financial_model.pkl'):
        """Initialize the feature completion system."""
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.training_data = None
        self.model_data = None
        self.company_profiles = None
        self.feature_relationships = None
        self.load_data()
        self.analyze_patterns()
    
    def load_data(self):
        """Load training data and model."""
        print("Loading training data and model...")
        
        # Load training data
        self.training_data = pd.read_csv(self.training_data_path)
        print(f"✅ Training data loaded: {len(self.training_data)} records")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        print(f"✅ Model loaded: expects {len(self.model_data['feature_cols'])} features")
    
    def analyze_patterns(self):
        """Analyze patterns in training data to build feature relationships."""
        print("Analyzing patterns in training data...")
        
        df = self.training_data.copy()
        
        # Clean and prepare data
        df = self._clean_training_data(df)
        
        # Create company profiles
        self.company_profiles = self._create_company_profiles(df)
        
        # Analyze feature relationships
        self.feature_relationships = self._analyze_feature_relationships(df)
        
        print("✅ Pattern analysis complete")
    
    def _clean_training_data(self, df):
        """Clean and prepare training data for analysis."""
        # Create time index
        df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
        df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
        df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
        df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
        
        # Calculate growth rates
        df['yoy_growth'] = df.groupby('id_company')['cARR'].pct_change(4)
        df['qoq_growth'] = df.groupby('id_company')['cARR'].pct_change(1)
        
        # Remove extreme outliers
        df = df[(df['yoy_growth'] >= -1.0) & (df['yoy_growth'] <= 5.0)]  # -100% to +500%
        df = df[(df['qoq_growth'] >= -0.5) & (df['qoq_growth'] <= 1.0)]  # -50% to +100%
        
        return df
    
    def _create_company_profiles(self, df):
        """Create company profiles based on size, growth, and characteristics."""
        print("Creating company profiles...")
        
        # Get latest data for each company
        latest_data = df.sort_values(['id_company', 'time_idx']).groupby('id_company').tail(1)
        
        profiles = []
        for _, row in latest_data.iterrows():
            profile = {
                'id_company': row['id_company'],
                'arr_size': row['cARR'],
                'yoy_growth': row['yoy_growth'],
                'qoq_growth': row['qoq_growth'],
                'headcount': row.get('Headcount (HC)', 0),
                'gross_margin': row.get('Gross Margin (in %)', 0),
                'sector': row.get('Sector', 'Unknown'),
                'country': row.get('Country', 'Unknown'),
                'currency': row.get('Currency', 'Unknown'),
                'target_customer': row.get('Target Customer', 'Unknown'),
                'deal_team': row.get('Deal Team', 'Unknown')
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _analyze_feature_relationships(self, df):
        """Analyze relationships between features to understand patterns."""
        print("Analyzing feature relationships...")
        
        relationships = {}
        
        # Analyze key financial relationships
        key_features = ['cARR', 'Headcount (HC)', 'Gross Margin (in %)', 'Sales & Marketing', 
                       'R&D', 'G&A', 'Cash Burn (OCF & ICF)', 'Net New ARR']
        
        for feature in key_features:
            if feature in df.columns:
                if feature != 'cARR':
                    corr = df[['cARR', feature]].corr().iloc[0, 1]
                    stats = {
                        'correlation_with_arr': corr,
                        'mean_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).mean(),
                        'median_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).median(),
                        'std_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).std()
                    }
                    relationships[feature] = stats
                    relationships[_sanitize_name(feature)] = stats
        
        # Analyze growth patterns
        relationships['growth_patterns'] = {
            'mean_yoy_growth': df['yoy_growth'].mean(),
            'median_yoy_growth': df['yoy_growth'].median(),
            'std_yoy_growth': df['yoy_growth'].std(),
            'mean_qoq_growth': df['qoq_growth'].mean(),
            'median_qoq_growth': df['qoq_growth'].median(),
            'std_qoq_growth': df['qoq_growth'].std()
        }
        
        return relationships
    
    def find_similar_companies(self, user_data):
        """Find companies similar to the user's company."""
        print("Finding similar companies...")
        
        # Extract user characteristics
        user_arr = user_data['cARR'].iloc[-1] if 'cARR' in user_data.columns else 1000000

        # Tier-based inputs often provide only 4 quarters (Q1-Q4), so `pct_change(4)` yields NaN.
        # Use a robust annualized growth proxy when YoY growth isn't available.
        user_growth = None
        if 'yoy_growth' in user_data.columns and len(user_data) > 0:
            user_growth = user_data['yoy_growth'].iloc[-1]
        if user_growth is None or (isinstance(user_growth, float) and np.isnan(user_growth)) or pd.isna(user_growth):
            user_growth = 0.2  # sensible fallback if we can't compute a proxy
            if 'cARR' in user_data.columns and len(user_data) >= 2:
                first_arr = user_data['cARR'].iloc[0]
                last_arr = user_data['cARR'].iloc[-1]
                if pd.notna(first_arr) and pd.notna(last_arr) and first_arr > 0:
                    # Q1→Q4 is 3 quarters of change; annualize to 4 quarters:
                    # annualized = (last/first)^(4/3) - 1
                    try:
                        user_growth = (float(last_arr) / float(first_arr)) ** (4 / 3) - 1
                    except Exception:
                        user_growth = 0.2

        user_headcount = user_data['Headcount (HC)'].iloc[-1] if 'Headcount (HC)' in user_data.columns else 50
        if user_headcount is None or pd.isna(user_headcount):
            user_headcount = 50
        
        # Calculate similarity scores
        profiles = self.company_profiles.copy()
        
        # Handle NaN values
        profiles = profiles.dropna(subset=['arr_size', 'yoy_growth', 'headcount'])
        
        # Use logarithmic similarity for ARR (companies can vary by orders of magnitude)
        profiles['arr_similarity'] = 1 / (1 + np.abs(np.log(profiles['arr_size'] + 1e-8) - np.log(user_arr + 1e-8)))
        profiles['growth_similarity'] = 1 / (1 + np.abs(profiles['yoy_growth'] - user_growth))
        profiles['size_similarity'] = 1 / (1 + np.abs(profiles['headcount'] - user_headcount) / (user_headcount + 1e-8))
        
        # Combined similarity score with better weighting
        profiles['similarity_score'] = (
            profiles['arr_similarity'] * 0.5 +  # ARR size is most important
            profiles['growth_similarity'] * 0.3 +  # Growth rate is important
            profiles['size_similarity'] * 0.2  # Headcount is less important
        )
        
        # Get top similar companies
        similar_companies = profiles.nlargest(50, 'similarity_score')  # Get more companies for better inference
        
        print(f"Found {len(similar_companies)} similar companies")
        if len(similar_companies) > 0:
            print(f"Top similarity score: {similar_companies['similarity_score'].iloc[0]:.3f}")
            print(f"User ARR: ${user_arr:,.0f}, Growth: {user_growth:.1%}")
            print(f"Most similar company: ARR=${similar_companies['arr_size'].iloc[0]:,.0f}, Growth={similar_companies['yoy_growth'].iloc[0]:.1%}")
        
        return similar_companies
    
    def infer_missing_features(self, user_data, similar_companies):
        """Infer missing features based on similar companies and patterns.

        The model's feature_cols use sanitized names (e.g. ``Gross_Margin__in___``).
        User data and the raw training CSV use the original names.  We build a
        reverse mapping so we can look up raw-named columns efficiently.
        """
        print("Inferring missing features...")

        latest_user = user_data.iloc[-1].copy()
        user_arr = latest_user['cARR']

        # Build sanitized → raw mapping for user data columns
        user_san_to_raw = {_sanitize_name(c): c for c in latest_user.index}

        # Get latest similar-company rows from raw training data
        similar_data = self.training_data[
            self.training_data['id_company'].isin(similar_companies['id_company'])
        ].copy()
        similar_data['Year'] = similar_data['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
        similar_data['Year'] = similar_data['Year'].apply(lambda x: x + 2000 if x < 100 else x)
        similar_data['Quarter Num'] = similar_data['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
        similar_data['time_idx'] = similar_data['Year'] * 4 + similar_data['Quarter Num']
        similar_latest = similar_data.sort_values(['id_company', 'time_idx']).groupby('id_company').tail(1)

        # Merge similarity scores so values and weights share the same row order.
        similar_latest = similar_latest.merge(
            similar_companies[['id_company', 'similarity_score']],
            on='id_company',
            how='left',
        )
        sim_weights = similar_latest['similarity_score'].values

        # Build sanitized → raw mapping for training CSV columns
        csv_san_to_raw = {_sanitize_name(c): c for c in similar_latest.columns}

        inferred_features = {}

        for feature in self.model_data['feature_cols']:
            # 1) Try user data (sanitized name → raw lookup)
            raw_user_col = user_san_to_raw.get(feature)
            if raw_user_col is not None and raw_user_col in latest_user.index:
                val = latest_user[raw_user_col]
                if not pd.isna(val):
                    inferred_features[feature] = val
                    continue

            # 2) Try similar companies (sanitized name → raw CSV column)
            raw_csv_col = csv_san_to_raw.get(feature)
            if raw_csv_col is not None and raw_csv_col in similar_latest.columns:
                col_series = similar_latest[raw_csv_col]
                if col_series.dtype in ['object', 'category']:
                    mode_value = col_series.mode()
                    if len(mode_value) > 0:
                        inferred_features[feature] = mode_value.iloc[0]
                    else:
                        inferred_features[feature] = self._get_default_value(feature, user_arr)
                else:
                    try:
                        values = col_series.values
                        if len(values) > 0 and not all(pd.isna(values)):
                            valid_mask = ~pd.isna(values)
                            valid_values = values[valid_mask].astype(float)
                            valid_weights = sim_weights[valid_mask]

                            if len(valid_values) > 0:
                                sorted_indices = np.argsort(valid_values)
                                sorted_values = valid_values[sorted_indices]
                                sorted_weights = valid_weights[sorted_indices]
                                cumsum_weights = np.cumsum(sorted_weights)
                                total_weight = cumsum_weights[-1]
                                median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
                                if median_idx < len(sorted_values):
                                    inferred_features[feature] = sorted_values[median_idx]
                                else:
                                    inferred_features[feature] = np.median(valid_values)
                            else:
                                inferred_features[feature] = self._get_default_value(feature, user_arr)
                        else:
                            inferred_features[feature] = self._get_default_value(feature, user_arr)
                    except Exception:
                        inferred_features[feature] = self._get_default_value(feature, user_arr)
                continue

            # 3) Fallback to defaults
            inferred_features[feature] = self._get_default_value(feature, user_arr)

        print(f"Inferred {len(inferred_features)} features")
        return inferred_features
    
    def _get_default_value(self, feature, user_arr):
        """Get default value for a feature based on patterns.

        ``feature`` arrives as a sanitized column name (e.g.
        ``Gross_Margin__in___``).  We look it up by both the sanitized
        and the raw name so the size-based defaults always apply.
        """
        # Use feature relationships to infer values
        if feature in self.feature_relationships:
            rel = self.feature_relationships[feature]
            if 'median_ratio_to_arr' in rel:
                return user_arr * rel['median_ratio_to_arr']
        
        # Defaults based on company size. user_arr is in THOUSANDS (training scale).
        if user_arr < 1000:  # Small company (< $1M ARR)
            raw_defaults = {
                'Gross Margin (in %)': 0.70,
                'Net Profit/Loss Margin (in %)': -25,
                'Headcount (HC)': max(1, int(user_arr / 100)),
                'Customers (EoP)': max(1, int(user_arr / 3)),
                'Sales & Marketing': user_arr * 0.5,
                'R&D': user_arr * 0.3,
                'G&A': user_arr * 0.2,
                'Cash Burn (OCF & ICF)': -user_arr * 0.4,
                'Expansion & Upsell': user_arr * 0.15,
                'Churn & Reduction': -user_arr * 0.08
            }
        elif user_arr < 10000:  # Medium company ($1M - $10M ARR)
            raw_defaults = {
                'Gross Margin (in %)': 0.75,
                'Net Profit/Loss Margin (in %)': -15,
                'Headcount (HC)': max(1, int(user_arr / 150)),
                'Customers (EoP)': max(1, int(user_arr / 5)),
                'Sales & Marketing': user_arr * 0.4,
                'R&D': user_arr * 0.25,
                'G&A': user_arr * 0.15,
                'Cash Burn (OCF & ICF)': -user_arr * 0.3,
                'Expansion & Upsell': user_arr * 0.1,
                'Churn & Reduction': -user_arr * 0.05
            }
        else:  # Large company (> $10M ARR)
            raw_defaults = {
                'Gross Margin (in %)': 0.80,
                'Net Profit/Loss Margin (in %)': -5,
                'Headcount (HC)': max(1, int(user_arr / 200)),
                'Customers (EoP)': max(1, int(user_arr / 8)),
                'Sales & Marketing': user_arr * 0.3,
                'R&D': user_arr * 0.2,
                'G&A': user_arr * 0.1,
                'Cash Burn (OCF & ICF)': -user_arr * 0.2,
                'Expansion & Upsell': user_arr * 0.08,
                'Churn & Reduction': -user_arr * 0.03
            }

        # Build a lookup that accepts both raw and sanitized names.
        defaults = {}
        for raw_name, value in raw_defaults.items():
            defaults[raw_name] = value
            defaults[_sanitize_name(raw_name)] = value
        
        return defaults.get(feature, 0)
    
    def _convert_categorical_to_numeric(self, feature_vector):
        """Convert categorical features to numeric codes."""
        # Define categorical features that need to be converted
        categorical_features = ['Currency', 'Sector', 'Target Customer', 'Country', 'Deal Team']
        
        for feature in categorical_features:
            if feature in feature_vector.columns:
                # Convert to numeric codes
                feature_vector[feature] = pd.Categorical(feature_vector[feature]).codes
        
        # Convert all remaining object columns to numeric
        for col in feature_vector.columns:
            if feature_vector[col].dtype == 'object':
                try:
                    feature_vector[col] = pd.to_numeric(feature_vector[col], errors='coerce')
                    feature_vector[col] = feature_vector[col].fillna(0)
                except Exception:
                    feature_vector[col] = 0
        
        return feature_vector
    
    def complete_features(self, user_data):
        """Complete missing features for user data."""
        print("Starting intelligent feature completion...")
        
        # Find similar companies
        similar_companies = self.find_similar_companies(user_data)
        
        # Infer missing features
        completed_features = self.infer_missing_features(user_data, similar_companies)
        
        # Create feature vector
        feature_vector = pd.DataFrame([completed_features])
        
        print("✅ Feature completion complete")
        return feature_vector, similar_companies
    
    def predict_with_completed_features(self, user_data):
        """Make predictions using completed features.

        Returns predictions in **percentage points** (e.g. 50 = 50% growth),
        matching the training target scale.
        """
        print("Making predictions with completed features...")

        feature_vector, similar_companies = self.complete_features(user_data)

        # Ensure all required features are present
        for feature in self.model_data['feature_cols']:
            if feature not in feature_vector.columns:
                feature_vector[feature] = 0

        # Convert categorical features to numeric codes
        feature_vector = self._convert_categorical_to_numeric(feature_vector)

        # Reorder columns to match model expectations
        feature_vector = feature_vector[self.model_data['feature_cols']]
        feature_vector = feature_vector.apply(pd.to_numeric, errors='coerce')

        # Apply saved preprocessing from training: clip_bounds then train_medians
        clip_bounds = self.model_data.get('clip_bounds', {})
        for col in feature_vector.columns:
            if col in clip_bounds:
                lo, hi = clip_bounds[col]
                feature_vector[col] = feature_vector[col].clip(lo, hi)

        train_medians = self.model_data.get('train_medians', {})
        for col in feature_vector.columns:
            if col in train_medians:
                feature_vector[col] = feature_vector[col].fillna(train_medians[col])
        feature_vector = feature_vector.fillna(0)

        # Make predictions
        model_pipeline = self.model_data['model_pipeline']
        predictions = model_pipeline.predict(feature_vector)[0]

        # Clip predictions to the target cap used during training
        target_cap = self.model_data.get('target_cap', 500)
        predictions = np.clip(predictions, -target_cap, target_cap)

        print(f"Raw predictions (YoY growth pct points): {predictions}")
        print(f"Predictions as growth %: {[f'{x:.1f}%' for x in predictions]}")

        return predictions, similar_companies, feature_vector
