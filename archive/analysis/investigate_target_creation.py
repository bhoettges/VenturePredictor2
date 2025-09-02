#!/usr/bin/env python3
"""
Investigate how target variables are created for each quarter.
"""

import pandas as pd
import numpy as np

def investigate_target_creation():
    """Investigate how target variables are created."""
    print("🔍 INVESTIGATING TARGET VARIABLE CREATION")
    print("=" * 60)
    
    # Load the training data
    df = pd.read_csv('202402_Copy_Fixed.csv')
    
    # Calculate quarterly growth rates
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Calculate quarterly growth
    df['QoQ_Growth'] = df.groupby('id_company')['cARR'].pct_change(1) * 100
    
    # Create target variables (same as in training)
    for i in range(1, 5):
        df[f'Target_Q{i}'] = df.groupby('id_company')['ARR YoY Growth (in %)'].shift(-i)
    
    # Filter for realistic growth rates
    df_clean = df[(df['QoQ_Growth'] >= -50) & (df['QoQ_Growth'] <= 200)].copy()
    
    print("📊 Target Variable Analysis:")
    print("Looking at companies with data for all 4 quarters ahead...")
    
    # Find companies with all 4 targets
    companies_with_all_targets = df_clean.dropna(subset=['Target_Q1', 'Target_Q2', 'Target_Q3', 'Target_Q4'])
    
    if len(companies_with_all_targets) > 0:
        print(f"Found {len(companies_with_all_targets)} company-quarters with all 4 targets")
        
        # Analyze target distributions
        print(f"\n📈 Target Variable Distributions:")
        for i in range(1, 5):
            target_col = f'Target_Q{i}'
            target_data = companies_with_all_targets[target_col]
            print(f"Target_Q{i}: Mean={target_data.mean():.1f}%, Median={target_data.median():.1f}%, 75th={target_data.quantile(0.75):.1f}%")
        
        # Show some examples
        print(f"\n📊 Sample Company Examples:")
        sample_companies = companies_with_all_targets.head(5)
        for _, row in sample_companies.iterrows():
            company = row['id_company']
            quarter = row['Financial Quarter']
            print(f"Company {company} - {quarter}:")
            print(f"  Target_Q1: {row['Target_Q1']:.1f}%")
            print(f"  Target_Q2: {row['Target_Q2']:.1f}%")
            print(f"  Target_Q3: {row['Target_Q3']:.1f}%")
            print(f"  Target_Q4: {row['Target_Q4']:.1f}%")
            print()
    
    else:
        print("❌ No companies found with all 4 targets - this might be the issue!")

def investigate_yoY_growth_calculation():
    """Investigate how YoY growth is calculated."""
    print(f"\n🔍 INVESTIGATING YOY GROWTH CALCULATION")
    print("=" * 60)
    
    # Load the training data
    df = pd.read_csv('202402_Copy_Fixed.csv')
    
    # Calculate quarterly growth rates
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
    df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Calculate quarterly growth
    df['QoQ_Growth'] = df.groupby('id_company')['cARR'].pct_change(1) * 100
    
    # Calculate YoY growth manually
    df['YoY_Growth_Manual'] = df.groupby('id_company')['cARR'].pct_change(periods=4) * 100
    
    # Compare with existing YoY growth
    print("📊 YoY Growth Comparison:")
    print(f"Existing YoY Growth: Mean={df['ARR YoY Growth (in %)'].mean():.1f}%, Median={df['ARR YoY Growth (in %)'].median():.1f}%")
    print(f"Manual YoY Growth: Mean={df['YoY_Growth_Manual'].mean():.1f}%, Median={df['YoY_Growth_Manual'].median():.1f}%")
    
    # Show some examples
    print(f"\n📊 Sample YoY Growth Examples:")
    sample_data = df[df['YoY_Growth_Manual'].notna()].head(5)
    for _, row in sample_data.iterrows():
        company = row['id_company']
        quarter = row['Financial Quarter']
        print(f"Company {company} - {quarter}:")
        print(f"  Existing YoY: {row['ARR YoY Growth (in %)']:.1f}%")
        print(f"  Manual YoY: {row['YoY_Growth_Manual']:.1f}%")
        print(f"  QoQ Growth: {row['QoQ_Growth']:.1f}%")
        print()

def main():
    """Main investigation function."""
    print("🔍 INVESTIGATING TARGET VARIABLE CREATION")
    print("=" * 80)
    
    # Investigate target creation
    investigate_target_creation()
    
    # Investigate YoY growth calculation
    investigate_yoY_growth_calculation()
    
    print(f"\n💡 INSIGHTS:")
    print(f"1. We need to understand how Target_Q1, Target_Q2, Target_Q3, Target_Q4 are created")
    print(f"2. The issue might be in the target variable creation, not the model training")
    print(f"3. If Target_Q1 is consistently higher than other targets, that explains the bias")
    print(f"4. We need to check if the YoY growth calculation is correct")

if __name__ == "__main__":
    main()


