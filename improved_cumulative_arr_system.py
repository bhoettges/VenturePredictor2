#!/usr/bin/env python3
"""
Improved Cumulative ARR Prediction System
========================================

A more robust system that predicts cumulative ARR and breaks it down into quarterly targets.
"""

import pandas as pd
import numpy as np
import joblib
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

class ImprovedCumulativeARRSystem:
    def __init__(self):
        self.model_data = None
        self.completion_system = None
        
    def load_models(self):
        """Load both the cumulative ARR model and feature completion system."""
        print("Loading models...")
        
        # Load cumulative ARR model
        self.model_data = joblib.load('cumulative_arr_model.pkl')
        
        # Load feature completion system
        self.completion_system = IntelligentFeatureCompletionSystem()
        
        print("✅ Models loaded successfully")
    
    def predict_cumulative_arr_2024(self, company_df):
        """Predict cumulative ARR for 2024 with improved logic."""
        print("Predicting cumulative ARR for 2024...")
        
        # Calculate current cumulative ARR
        company_df['Cumulative_ARR'] = company_df['cARR'].cumsum()
        current_cumulative_arr = company_df['Cumulative_ARR'].iloc[-1]
        
        print(f"Current Cumulative ARR (Q4 2023): ${current_cumulative_arr:,.0f}")
        
        # Use feature completion to get rich features
        yoy_predictions, similar_companies, feature_vector = self.completion_system.predict_with_completed_features(company_df)
        
        # Prepare features for cumulative ARR model
        feature_cols = self.model_data['feature_cols']
        
        # Create feature vector
        last_quarter = company_df.iloc[-1]
        feature_vector_cumulative = []
        
        for col in feature_cols:
            if col in last_quarter:
                feature_vector_cumulative.append(last_quarter[col])
            else:
                feature_vector_cumulative.append(0)
        
        feature_vector_cumulative = np.array(feature_vector_cumulative).reshape(1, -1)
        
        # Scale and select features
        scaler = self.model_data['scaler']
        feature_selector = self.model_data['feature_selector']
        
        X_scaled = scaler.transform(feature_vector_cumulative)
        X_selected = feature_selector.transform(X_scaled)
        
        # Make cumulative ARR predictions
        models = self.model_data['models']
        target_cols = self.model_data['target_cols']
        
        cumulative_growth_predictions = {}
        for target_col in target_cols:
            pred = models[target_col].predict(X_selected)[0]
            cumulative_growth_predictions[target_col] = pred
        
        # Calculate cumulative ARR for 2024
        cumulative_arr_2024 = {}
        for i, target_col in enumerate(target_cols):
            quarter = f"Q{i+1}"
            growth = cumulative_growth_predictions[target_col]
            cumulative_arr_2024[quarter] = current_cumulative_arr * (1 + growth)
        
        return cumulative_arr_2024, cumulative_growth_predictions, yoy_predictions
    
    def create_quarterly_breakdown(self, cumulative_arr_2024, current_cumulative_arr):
        """Create quarterly ARR breakdown from cumulative predictions."""
        print("Creating quarterly ARR breakdown...")
        
        # Calculate quarterly ARR contributions
        quarterly_arr = {}
        quarterly_arr['Q1'] = cumulative_arr_2024['Q1'] - current_cumulative_arr
        quarterly_arr['Q2'] = cumulative_arr_2024['Q2'] - cumulative_arr_2024['Q1']
        quarterly_arr['Q3'] = cumulative_arr_2024['Q3'] - cumulative_arr_2024['Q2']
        quarterly_arr['Q4'] = cumulative_arr_2024['Q4'] - cumulative_arr_2024['Q3']
        
        # Ensure no negative quarterly contributions (business reality)
        for quarter, arr in quarterly_arr.items():
            if arr < 0:
                print(f"⚠️  Warning: {quarter} 2024 has negative ARR contribution (${arr:,.0f})")
                # Set to minimum positive value
                quarterly_arr[quarter] = max(1000, current_cumulative_arr * 0.01)  # 1% of current ARR
        
        # Recalculate cumulative ARR with corrected quarterly contributions
        corrected_cumulative = {}
        corrected_cumulative['Q1'] = current_cumulative_arr + quarterly_arr['Q1']
        corrected_cumulative['Q2'] = corrected_cumulative['Q1'] + quarterly_arr['Q2']
        corrected_cumulative['Q3'] = corrected_cumulative['Q2'] + quarterly_arr['Q3']
        corrected_cumulative['Q4'] = corrected_cumulative['Q3'] + quarterly_arr['Q4']
        
        return quarterly_arr, corrected_cumulative
    
    def predict_with_breakdown(self, company_df):
        """Main prediction function with quarterly breakdown."""
        print("IMPROVED CUMULATIVE ARR PREDICTION SYSTEM")
        print("=" * 60)
        
        # Load models if not already loaded
        if self.model_data is None:
            self.load_models()
        
        # Get cumulative ARR predictions
        cumulative_arr_2024, cumulative_growth_predictions, yoy_predictions = self.predict_cumulative_arr_2024(company_df)
        
        # Get current cumulative ARR
        current_cumulative_arr = company_df['cARR'].cumsum().iloc[-1]
        
        # Create quarterly breakdown
        quarterly_arr, corrected_cumulative = self.create_quarterly_breakdown(cumulative_arr_2024, current_cumulative_arr)
        
        # Display results
        print(f"\n" + "=" * 60)
        print("CUMULATIVE ARR PREDICTIONS")
        print("=" * 60)
        print(f"Starting point (Q4 2023): ${current_cumulative_arr:,.0f}")
        print()
        
        for quarter, cumulative_arr in corrected_cumulative.items():
            growth = (cumulative_arr - current_cumulative_arr) / current_cumulative_arr * 100
            print(f"{quarter} 2024: ${cumulative_arr:,.0f} ({growth:.1f}% cumulative growth)")
        
        print(f"\n" + "=" * 60)
        print("QUARTERLY ARR BREAKDOWN")
        print("=" * 60)
        
        for quarter, arr in quarterly_arr.items():
            print(f"{quarter} 2024: ${arr:,.0f}")
        
        # Summary
        total_2024_arr = corrected_cumulative['Q4']
        total_growth = (total_2024_arr - current_cumulative_arr) / current_cumulative_arr * 100
        avg_quarterly = np.mean(list(quarterly_arr.values()))
        
        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Starting Cumulative ARR (Q4 2023): ${current_cumulative_arr:,.0f}")
        print(f"Ending Cumulative ARR (Q4 2024): ${total_2024_arr:,.0f}")
        print(f"Total Growth: {total_growth:.1f}%")
        print(f"Average Quarterly Contribution: ${avg_quarterly:,.0f}")
        
        return {
            'cumulative_arr_2024': corrected_cumulative,
            'quarterly_arr': quarterly_arr,
            'cumulative_growth_predictions': cumulative_growth_predictions,
            'yoy_predictions': yoy_predictions
        }

def test_improved_system():
    """Test the improved cumulative ARR system."""
    # Load the new company data
    company_df = pd.read_csv('test_company_new.csv')
    
    # Map columns to expected format
    company_df['Financial Quarter'] = company_df['Quarter']
    company_df['cARR'] = company_df['ARR_End_of_Quarter']
    company_df['Headcount (HC)'] = company_df['Headcount']
    company_df['Gross Margin (in %)'] = company_df['Gross_Margin_Percent']
    company_df['id_company'] = 'test_company_new'
    
    # Initialize and test the system
    system = ImprovedCumulativeARRSystem()
    results = system.predict_with_breakdown(company_df)
    
    return results

if __name__ == "__main__":
    results = test_improved_system()

