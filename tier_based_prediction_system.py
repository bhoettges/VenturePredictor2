#!/usr/bin/env python3
"""
Tier-Based Prediction System
============================

Tier 1 (MUST PROVIDE): Q1-Q4 ARR, Headcount, Sector
Tier 2 (Advanced Analysis): Gross Margin, Sales & Marketing, Cash Burn, Churn Rate, Customers
"""

import pandas as pd
import numpy as np
from intelligent_feature_completion_system import IntelligentFeatureCompletionSystem

class TierBasedPredictionSystem:
    def __init__(self):
        self.completion_system = IntelligentFeatureCompletionSystem()
        
    def create_company_dataframe(self, tier1_data, tier2_data=None):
        """Create company dataframe from tier-based input."""
        print("Creating company dataframe from tier-based input...")
        
        # Tier 1 data (required)
        quarters = ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
        arr_values = [tier1_data['q1_arr'], tier1_data['q2_arr'], 
                     tier1_data['q3_arr'], tier1_data['q4_arr']]
        
        # Create base dataframe
        company_df = pd.DataFrame({
            'Quarter': quarters,
            'ARR_End_of_Quarter': arr_values,
            'Financial Quarter': quarters,
            'cARR': arr_values,
            'Headcount (HC)': [tier1_data['headcount']] * 4,
            'Sector': [tier1_data['sector']] * 4,
            'id_company': 'user_company'
        })
        
        # Calculate growth rates
        company_df['yoy_growth'] = company_df['cARR'].pct_change(4)
        company_df['qoq_growth'] = company_df['cARR'].pct_change(1)
        
        # Add Tier 2 data if provided
        if tier2_data:
            print("Adding Tier 2 advanced metrics...")
            
            # Financial metrics
            company_df['Gross Margin (in %)'] = tier2_data.get('gross_margin', 75)
            company_df['Sales & Marketing'] = tier2_data.get('sales_marketing', company_df['cARR'].iloc[-1] * 0.4)
            company_df['Cash Burn (OCF & ICF)'] = tier2_data.get('cash_burn', -company_df['cARR'].iloc[-1] * 0.3)
            company_df['Customers (EoP)'] = tier2_data.get('customers', int(company_df['cARR'].iloc[-1] / 5000))
            
            # Calculate churn and expansion from rates
            churn_rate = tier2_data.get('churn_rate', 0.05)  # 5% default
            expansion_rate = tier2_data.get('expansion_rate', 0.10)  # 10% default
            
            company_df['Churn & Reduction'] = -company_df['cARR'] * churn_rate
            company_df['Expansion & Upsell'] = company_df['cARR'] * expansion_rate
            
            print(f"  ✅ Gross Margin: {tier2_data.get('gross_margin', 75)}%")
            print(f"  ✅ Sales & Marketing: ${tier2_data.get('sales_marketing', company_df['cARR'].iloc[-1] * 0.4):,.0f}")
            print(f"  ✅ Cash Burn: ${tier2_data.get('cash_burn', -company_df['cARR'].iloc[-1] * 0.3):,.0f}")
            print(f"  ✅ Customers: {tier2_data.get('customers', int(company_df['cARR'].iloc[-1] / 5000))}")
            print(f"  ✅ Churn Rate: {churn_rate*100:.1f}%")
            print(f"  ✅ Expansion Rate: {expansion_rate*100:.1f}%")
        else:
            print("Using intelligent defaults for Tier 2 metrics...")
            # Use intelligent defaults
            last_arr = company_df['cARR'].iloc[-1]
            company_df['Gross Margin (in %)'] = 75
            company_df['Sales & Marketing'] = last_arr * 0.4
            company_df['Cash Burn (OCF & ICF)'] = -last_arr * 0.3
            company_df['Customers (EoP)'] = int(last_arr / 5000)
            company_df['Churn & Reduction'] = -last_arr * 0.05
            company_df['Expansion & Upsell'] = last_arr * 0.10
        
        return company_df
    
    def predict_with_tiers(self, tier1_data, tier2_data=None):
        """Make predictions using tier-based input system."""
        print("TIER-BASED PREDICTION SYSTEM")
        print("=" * 60)
        
        # Display input summary
        print("TIER 1 INPUT (Required):")
        print(f"  Q1 2023 ARR: ${tier1_data['q1_arr']:,.0f}")
        print(f"  Q2 2023 ARR: ${tier1_data['q2_arr']:,.0f}")
        print(f"  Q3 2023 ARR: ${tier1_data['q3_arr']:,.0f}")
        print(f"  Q4 2023 ARR: ${tier1_data['q4_arr']:,.0f}")
        print(f"  Headcount: {tier1_data['headcount']}")
        print(f"  Sector: {tier1_data['sector']}")
        
        if tier2_data:
            print("\nTIER 2 INPUT (Advanced Analysis):")
            print(f"  Gross Margin: {tier2_data.get('gross_margin', 75)}%")
            print(f"  Sales & Marketing: ${tier2_data.get('sales_marketing', 0):,.0f}")
            print(f"  Cash Burn: ${tier2_data.get('cash_burn', 0):,.0f}")
            print(f"  Customers: {tier2_data.get('customers', 0)}")
            print(f"  Churn Rate: {tier2_data.get('churn_rate', 0.05)*100:.1f}%")
        else:
            print("\nTIER 2: Using intelligent defaults")
        
        # Create company dataframe
        company_df = self.create_company_dataframe(tier1_data, tier2_data)
        
        # Get predictions using the enhanced model
        yoy_predictions, similar_companies, feature_vector = self.completion_system.predict_with_completed_features(company_df)
        
        print(f"\nMODEL PREDICTIONS:")
        print("-" * 40)
        for i, pred in enumerate(yoy_predictions):
            print(f"  Q{i+1} 2024: {pred*100:.1f}% YoY growth")
        
        # Calculate absolute ARR predictions with QoQ growth
        q4_2023_arr = company_df.iloc[3]['cARR']  # Starting point for QoQ calculations
        
        print(f"\nARR PREDICTIONS WITH CONFIDENCE INTERVALS:")
        print("-" * 60)
        
        predictions = []
        current_arr = q4_2023_arr  # Start from Q4 2023
        
        for i, (quarter, yoy_growth) in enumerate(zip(
            ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
            yoy_predictions
        )):
            # Calculate target ARR based on YoY growth from same quarter previous year
            base_quarter_arr = company_df.iloc[i]['cARR']  # Q1 2023, Q2 2023, etc.
            target_arr = base_quarter_arr * (1 + yoy_growth)
            
            # Calculate QoQ growth from previous quarter
            qoq_growth = ((target_arr - current_arr) / current_arr) * 100 if current_arr > 0 else 0
            
            # Calculate confidence intervals
            pessimistic_arr = target_arr * 0.9  # -10%
            optimistic_arr = target_arr * 1.1   # +10%
            
            predictions.append({
                'Quarter': quarter,
                'ARR': target_arr,
                'Pessimistic_ARR': pessimistic_arr,
                'Optimistic_ARR': optimistic_arr,
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100,
                'QoQ_Growth_Percent': qoq_growth
            })
            
            print(f"{quarter}: ${target_arr:,.0f} (${pessimistic_arr:,.0f} - ${optimistic_arr:,.0f}) ({qoq_growth:+.1f}% QoQ)")
            current_arr = target_arr  # Update for next quarter
        
        return predictions, company_df

def test_tier_system():
    """Test the tier-based system with sample data."""
    
    # Tier 1 data (required)
    tier1_data = {
        'q1_arr': 1000000,    # $1M
        'q2_arr': 1400000,    # $1.4M  
        'q3_arr': 2000000,    # $2M
        'q4_arr': 2800000,    # $2.8M
        'headcount': 102,
        'sector': 'SaaS'
    }
    
    # Tier 2 data (optional - for advanced analysis)
    tier2_data = {
        'gross_margin': 75,
        'sales_marketing': 1200000,  # $1.2M
        'cash_burn': -800000,        # -$800K
        'customers': 200,
        'churn_rate': 0.05,          # 5%
        'expansion_rate': 0.15       # 15%
    }
    
    # Initialize system
    tier_system = TierBasedPredictionSystem()
    
    # Test with Tier 1 only
    print("TEST 1: Tier 1 Only (Required Input)")
    print("=" * 50)
    predictions_tier1, company_df_tier1 = tier_system.predict_with_tiers(tier1_data)
    
    print("\n" + "=" * 60)
    print("TEST 2: Tier 1 + Tier 2 (Advanced Analysis)")
    print("=" * 50)
    predictions_tier2, company_df_tier2 = tier_system.predict_with_tiers(tier1_data, tier2_data)
    
    # Compare results
    print(f"\n" + "=" * 60)
    print("COMPARISON: Tier 1 vs Tier 1+2")
    print("=" * 60)
    print(f"{'Quarter':<12} {'Tier 1 Only':<20} {'Tier 1+2':<20} {'Difference':<15}")
    print("-" * 70)
    
    for i, (pred1, pred2) in enumerate(zip(predictions_tier1, predictions_tier2)):
        quarter = pred1['Quarter']
        arr1 = pred1['ARR']
        arr2 = pred2['ARR']
        diff = arr2 - arr1
        diff_pct = (diff / arr1) * 100 if arr1 != 0 else 0
        
        print(f"{quarter:<12} ${arr1:<19,.0f} ${arr2:<19,.0f} ${diff:<14,.0f} ({diff_pct:+.1f}%)")
    
    return predictions_tier1, predictions_tier2

if __name__ == "__main__":
    predictions_tier1, predictions_tier2 = test_tier_system()

