#!/usr/bin/env python3
"""
Fix the YoY growth calculation in the training data.
The current 'ARR YoY Growth (in %)' field appears to be incorrectly calculated.
"""

import pandas as pd
import numpy as np

def fix_yoy_calculation():
    """Fix the YoY growth calculation in the training data."""
    print("🔧 FIXING YOY GROWTH CALCULATION IN TRAINING DATA")
    print("=" * 60)
    
    # Load the training data
    df = pd.read_csv('202402_Copy.csv')
    print(f"📊 Loaded {len(df)} rows of training data")
    
    # Create time_idx for sorting
    df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
    df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
    
    # Sort by company and time
    df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
    
    # Calculate correct YoY growth
    print("📈 Calculating correct YoY growth...")
    df['ARR YoY Growth (in %)'] = np.nan  # Reset the field
    
    for company_id in df['id_company'].unique():
        company_data = df[df['id_company'] == company_id].copy()
        
        for _, row in company_data.iterrows():
            # Find same quarter from previous year
            prev_year_data = company_data[
                (company_data['Year'] == row['Year'] - 1) & 
                (company_data['Quarter Num'] == row['Quarter Num'])
            ]
            
            if len(prev_year_data) > 0 and prev_year_data.iloc[0]['cARR'] > 0:
                prev_arr = prev_year_data.iloc[0]['cARR']
                yoy_growth = ((row['cARR'] - prev_arr) / prev_arr) * 100
                df.loc[row.name, 'ARR YoY Growth (in %)'] = yoy_growth
    
    # Also calculate Revenue YoY Growth correctly
    print("📈 Calculating correct Revenue YoY growth...")
    df['Revenue YoY Growth (in %)'] = np.nan  # Reset the field
    
    for company_id in df['id_company'].unique():
        company_data = df[df['id_company'] == company_id].copy()
        
        for _, row in company_data.iterrows():
            # Find same quarter from previous year
            prev_year_data = company_data[
                (company_data['Year'] == row['Year'] - 1) & 
                (company_data['Quarter Num'] == row['Quarter Num'])
            ]
            
            if len(prev_year_data) > 0 and prev_year_data.iloc[0]['cARR'] > 0:
                prev_revenue = prev_year_data.iloc[0]['cARR']  # Assuming revenue = ARR for now
                current_revenue = row['cARR']
                yoy_growth = ((current_revenue - prev_revenue) / prev_revenue) * 100
                df.loc[row.name, 'Revenue YoY Growth (in %)'] = yoy_growth
    
    # Show statistics of the corrected data
    print("\n📊 CORRECTED YOY GROWTH STATISTICS:")
    yoy_growth = df['ARR YoY Growth (in %)'].dropna()
    print(f"Mean YoY Growth: {yoy_growth.mean():.2f}%")
    print(f"Median YoY Growth: {yoy_growth.median():.2f}%")
    print(f"90th Percentile: {yoy_growth.quantile(0.9):.2f}%")
    print(f"95th Percentile: {yoy_growth.quantile(0.95):.2f}%")
    print(f"99th Percentile: {yoy_growth.quantile(0.99):.2f}%")
    print(f"Max YoY Growth: {yoy_growth.max():.2f}%")
    
    # Show some examples
    print("\n🚀 HIGH GROWTH EXAMPLES (CORRECTED):")
    high_growth = df[df['ARR YoY Growth (in %)'] > 100].head(10)
    for _, row in high_growth.iterrows():
        print(f"  Company {row['id_company']:3.0f} | {row['Financial Quarter']:8s} | YoY: {row['ARR YoY Growth (in %)']:6.1f}% | ARR: ${row['cARR']:8,.0f}")
    
    # Save the corrected data
    output_file = '202402_Copy_Fixed.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Corrected data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    fixed_df = fix_yoy_calculation()


