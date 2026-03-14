#!/usr/bin/env python3
"""
Test how the model handles declining companies
"""

from tier_based_prediction_system import TierBasedPredictionSystem
import pandas as pd

def test_declining_company():
    """Test with a clearly declining company."""
    
    print("=" * 80)
    print("TEST: DECLINING COMPANY SCENARIO")
    print("=" * 80)
    
    # Severely declining company
    tier1_data = {
        'q1_arr': 2000000,    # $2M
        'q2_arr': 1500000,    # $1.5M (-25%)
        'q3_arr': 1000000,    # $1M (-33%)
        'q4_arr': 500000,     # $500K (-50%)
        'headcount': 50,
        'sector': 'Data & Analytics'
    }
    
    print("\n📉 DECLINING COMPANY INPUT:")
    print(f"  Q1 2023: ${tier1_data['q1_arr']:,} → Q4 2023: ${tier1_data['q4_arr']:,}")
    print(f"  Total Decline: {((tier1_data['q4_arr'] - tier1_data['q1_arr']) / tier1_data['q1_arr'] * 100):.1f}%")
    print(f"  Q1→Q2: {((tier1_data['q2_arr'] - tier1_data['q1_arr']) / tier1_data['q1_arr'] * 100):.1f}%")
    print(f"  Q2→Q3: {((tier1_data['q3_arr'] - tier1_data['q2_arr']) / tier1_data['q2_arr'] * 100):.1f}%")
    print(f"  Q3→Q4: {((tier1_data['q4_arr'] - tier1_data['q3_arr']) / tier1_data['q3_arr'] * 100):.1f}%")
    print(f"  Sector: {tier1_data['sector']}")
    print(f"  Headcount: {tier1_data['headcount']}")
    
    # Initialize system
    tier_system = TierBasedPredictionSystem()
    
    # Get predictions
    predictions, company_df = tier_system.predict_with_tiers(tier1_data)
    
    print("\n" + "=" * 80)
    print("ANALYSIS: Does the model recognize the decline?")
    print("=" * 80)
    
    # Check if predictions show growth or decline
    for pred in predictions:
        quarter = pred['Quarter']
        arr = pred['ARR']
        yoy_growth = pred['YoY_Growth_Percent']
        
        # Compare to Q4 2023 (most recent quarter)
        qoq_from_q4 = ((arr - tier1_data['q4_arr']) / tier1_data['q4_arr']) * 100
        
        print(f"\n{quarter}:")
        print(f"  Predicted ARR: ${arr:,.0f}")
        print(f"  YoY Growth: {yoy_growth:.1f}%")
        print(f"  Growth from Q4 2023: {qoq_from_q4:+.1f}%")
        
        if arr > tier1_data['q4_arr']:
            print(f"  ⚠️  PROBLEM: Predicting growth despite 75% decline in 2023!")
        else:
            print(f"  ✅ Correctly predicting continued decline")

def test_growing_company():
    """Test with a growing company for comparison."""
    
    print("\n" + "=" * 80)
    print("TEST: GROWING COMPANY SCENARIO (for comparison)")
    print("=" * 80)
    
    # Growing company
    tier1_data = {
        'q1_arr': 1000000,    # $1M
        'q2_arr': 1400000,    # $1.4M (+40%)
        'q3_arr': 2000000,    # $2M (+43%)
        'q4_arr': 2800000,    # $2.8M (+40%)
        'headcount': 100,
        'sector': 'Data & Analytics'
    }
    
    print("\n📈 GROWING COMPANY INPUT:")
    print(f"  Q1 2023: ${tier1_data['q1_arr']:,} → Q4 2023: ${tier1_data['q4_arr']:,}")
    print(f"  Total Growth: {((tier1_data['q4_arr'] - tier1_data['q1_arr']) / tier1_data['q1_arr'] * 100):.1f}%")
    
    # Initialize system
    tier_system = TierBasedPredictionSystem()
    
    # Get predictions
    predictions, company_df = tier_system.predict_with_tiers(tier1_data)
    
    print("\n📊 Predictions for growing company:")
    for pred in predictions:
        print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} ({pred['YoY_Growth_Percent']:.1f}% YoY)")

def analyze_training_data():
    """Analyze what percentage of training data shows growth vs decline."""
    
    print("\n" + "=" * 80)
    print("TRAINING DATA ANALYSIS")
    print("=" * 80)
    
    df = pd.read_csv('202402_Copy.csv')
    
    # Calculate QoQ growth
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    
    df = df.sort_values(['id_company', 'Year', 'Quarter Num'])
    df['qoq_growth'] = df.groupby('id_company')['cARR'].pct_change(1)
    
    # Filter valid growth values
    valid_growth = df['qoq_growth'].dropna()
    
    growing = (valid_growth > 0).sum()
    declining = (valid_growth < 0).sum()
    flat = (valid_growth == 0).sum()
    
    print(f"\nQuarter-over-Quarter Growth Distribution:")
    print(f"  Growing quarters: {growing:,} ({growing/len(valid_growth)*100:.1f}%)")
    print(f"  Declining quarters: {declining:,} ({declining/len(valid_growth)*100:.1f}%)")
    print(f"  Flat quarters: {flat:,} ({flat/len(valid_growth)*100:.1f}%)")
    
    print(f"\nAverage QoQ Growth: {valid_growth.mean()*100:.2f}%")
    print(f"Median QoQ Growth: {valid_growth.median()*100:.2f}%")
    
    # Check for severely declining companies (like our test case)
    severe_decline = (valid_growth < -0.25).sum()  # More than 25% decline
    print(f"\nSevere declines (>25% QoQ): {severe_decline:,} ({severe_decline/len(valid_growth)*100:.1f}%)")
    
    print("\n⚠️  CONCLUSION:")
    if declining < growing * 0.3:  # Less than 30% declining
        print("  Training data is heavily biased toward GROWTH!")
        print("  Model likely learned to always predict growth.")
        print("  This explains why declining companies get growth predictions.")

if __name__ == "__main__":
    # Test declining company
    test_declining_company()
    
    # Test growing company for comparison
    test_growing_company()
    
    # Analyze training data bias
    analyze_training_data()

