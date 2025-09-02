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

import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IntelligentFeatureCompletionSystem:
    """Advanced feature completion system using company profiling and industry patterns."""
    
    def __init__(self, training_data_path='202402_Copy.csv', model_path='lightgbm_financial_model.pkl'):
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
                # Calculate correlations with ARR
                if feature != 'cARR':
                    corr = df[['cARR', feature]].corr().iloc[0, 1]
                    relationships[feature] = {
                        'correlation_with_arr': corr,
                        'mean_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).mean(),
                        'median_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).median(),
                        'std_ratio_to_arr': (df[feature] / (df['cARR'] + 1e-8)).std()
                    }
        
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
        user_growth = user_data['yoy_growth'].iloc[-1] if 'yoy_growth' in user_data.columns else 0.2
        user_headcount = user_data['Headcount (HC)'].iloc[-1] if 'Headcount (HC)' in user_data.columns else 50
        
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
        """Infer missing features based on similar companies and patterns."""
        print("Inferring missing features...")
        
        # Get the latest user data
        latest_user = user_data.iloc[-1].copy()
        user_arr = latest_user['cARR']
        
        # Initialize feature dictionary
        inferred_features = {}
        
        # Use similar companies to infer features
        similar_data = self.training_data[
            self.training_data['id_company'].isin(similar_companies['id_company'])
        ]
        
        # Get latest data for similar companies
        # Sort by company and year/quarter to get latest data
        similar_data['Year'] = similar_data['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
        similar_data['Year'] = similar_data['Year'].apply(lambda x: x + 2000 if x < 100 else x)
        similar_data['Quarter Num'] = similar_data['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
        similar_data['time_idx'] = similar_data['Year'] * 4 + similar_data['Quarter Num']
        
        similar_latest = similar_data.sort_values(['id_company', 'time_idx']).groupby('id_company').tail(1)
        
        # Infer features based on similar companies with weighted averages
        for feature in self.model_data['feature_cols']:
            if feature in latest_user and not pd.isna(latest_user[feature]):
                # Use user's actual value
                inferred_features[feature] = latest_user[feature]
            elif feature in similar_latest.columns:
                # Handle different data types
                if similar_latest[feature].dtype in ['object', 'category']:
                    # For categorical features, use mode (most common value)
                    mode_value = similar_latest[feature].mode()
                    if len(mode_value) > 0:
                        inferred_features[feature] = mode_value.iloc[0]
                    else:
                        inferred_features[feature] = self._get_default_value(feature, user_arr)
                else:
                    # For numeric features, use weighted median based on similarity
                    try:
                        # Get similarity scores for the similar companies
                        similar_ids = similar_latest['id_company'].values
                        similarity_scores = similar_companies[similar_companies['id_company'].isin(similar_ids)]['similarity_score'].values
                        
                        # Use weighted median (weighted by similarity)
                        values = similar_latest[feature].values
                        if len(values) > 0 and not all(pd.isna(values)):
                            # Remove NaN values and corresponding weights
                            valid_mask = ~pd.isna(values)
                            valid_values = values[valid_mask]
                            valid_weights = similarity_scores[valid_mask]
                            
                            if len(valid_values) > 0:
                                # Calculate weighted median
                                sorted_indices = np.argsort(valid_values)
                                sorted_values = valid_values[sorted_indices]
                                sorted_weights = valid_weights[sorted_indices]
                                cumsum_weights = np.cumsum(sorted_weights)
                                total_weight = cumsum_weights[-1]
                                
                                # Find median
                                median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
                                if median_idx < len(sorted_values):
                                    inferred_features[feature] = sorted_values[median_idx]
                                else:
                                    inferred_features[feature] = np.median(valid_values)
                            else:
                                inferred_features[feature] = self._get_default_value(feature, user_arr)
                        else:
                            inferred_features[feature] = self._get_default_value(feature, user_arr)
                    except:
                        inferred_features[feature] = self._get_default_value(feature, user_arr)
            else:
                # Use default value
                inferred_features[feature] = self._get_default_value(feature, user_arr)
        
        print(f"Inferred {len(inferred_features)} features")
        return inferred_features
    
    def _get_default_value(self, feature, user_arr):
        """Get default value for a feature based on patterns."""
        # Use feature relationships to infer values
        if feature in self.feature_relationships:
            rel = self.feature_relationships[feature]
            if 'median_ratio_to_arr' in rel:
                return user_arr * rel['median_ratio_to_arr']
        
        # More realistic industry-standard defaults based on company size
        if user_arr < 1000000:  # Small company (< $1M ARR)
            defaults = {
                'Gross Margin (in %)': 70,
                'Net Profit/Loss Margin (in %)': -25,
                'Headcount (HC)': max(1, int(user_arr / 100000)),
                'Customers (EoP)': max(1, int(user_arr / 3000)),
                'Sales & Marketing': user_arr * 0.5,
                'R&D': user_arr * 0.3,
                'G&A': user_arr * 0.2,
                'Cash Burn (OCF & ICF)': -user_arr * 0.4,
                'Expansion & Upsell': user_arr * 0.15,
                'Churn & Reduction': -user_arr * 0.08
            }
        elif user_arr < 10000000:  # Medium company ($1M - $10M ARR)
            defaults = {
                'Gross Margin (in %)': 75,
                'Net Profit/Loss Margin (in %)': -15,
                'Headcount (HC)': max(1, int(user_arr / 150000)),
                'Customers (EoP)': max(1, int(user_arr / 5000)),
                'Sales & Marketing': user_arr * 0.4,
                'R&D': user_arr * 0.25,
                'G&A': user_arr * 0.15,
                'Cash Burn (OCF & ICF)': -user_arr * 0.3,
                'Expansion & Upsell': user_arr * 0.1,
                'Churn & Reduction': -user_arr * 0.05
            }
        else:  # Large company (> $10M ARR)
            defaults = {
                'Gross Margin (in %)': 80,
                'Net Profit/Loss Margin (in %)': -5,
                'Headcount (HC)': max(1, int(user_arr / 200000)),
                'Customers (EoP)': max(1, int(user_arr / 8000)),
                'Sales & Marketing': user_arr * 0.3,
                'R&D': user_arr * 0.2,
                'G&A': user_arr * 0.1,
                'Cash Burn (OCF & ICF)': -user_arr * 0.2,
                'Expansion & Upsell': user_arr * 0.08,
                'Churn & Reduction': -user_arr * 0.03
            }
        
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
                except:
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
        """Make predictions using completed features."""
        print("Making predictions with completed features...")
        
        # Complete features
        feature_vector, similar_companies = self.complete_features(user_data)
        
        # Ensure all required features are present
        for feature in self.model_data['feature_cols']:
            if feature not in feature_vector.columns:
                feature_vector[feature] = 0
        
        # Convert categorical features to numeric codes
        feature_vector = self._convert_categorical_to_numeric(feature_vector)
        
        # Reorder columns to match model expectations
        feature_vector = feature_vector[self.model_data['feature_cols']]
        
        # Make predictions
        model_pipeline = self.model_data['model_pipeline']
        predictions = model_pipeline.predict(feature_vector)[0]
        
        print(f"Raw predictions: {predictions}")
        print(f"Predictions as percentages: {[f'{x*100:.1f}%' for x in predictions]}")
        
        return predictions, similar_companies, feature_vector

def main():
    """Test the intelligent feature completion system."""
    print("Intelligent Feature Completion System")
    print("=" * 50)
    
    # Initialize system
    completion_system = IntelligentFeatureCompletionSystem()
    
    # Load test company data
    print("\nLoading test company data...")
    company_df = pd.read_csv('test_company_2024.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_2024'
    
    # Calculate growth rates
    company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
    company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
    
    print("Company Data:")
    print(company_df[['Financial Quarter', 'cARR', 'Headcount (HC)', 'Gross Margin (in %)']])
    
    # Make predictions with completed features
    predictions, similar_companies, feature_vector = completion_system.predict_with_completed_features(company_df)
    
    # Display results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    current_arr = company_df['cARR'].iloc[-1]
    print(f"Current ARR: ${current_arr:,.0f}")
    
    for i, growth_rate in enumerate(predictions):
        future_arr = current_arr * (1 + growth_rate)
        quarter = f"Q{i+1} 2024"
        print(f"{quarter}: ${future_arr:,.0f} (Growth: {growth_rate*100:.1f}%)")
    
    print("\nSimilar Companies Found:")
    print("=" * 25)
    for i, (_, company) in enumerate(similar_companies.head(5).iterrows()):
        print(f"{i+1}. Company {company['id_company']}: "
              f"ARR=${company['arr_size']:,.0f}, "
              f"Growth={company['yoy_growth']*100:.1f}%, "
              f"Similarity={company['similarity_score']:.3f}")

if __name__ == "__main__":
    main()
