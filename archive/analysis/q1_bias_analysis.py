#!/usr/bin/env python3
"""
Simple analysis to understand the Q1 bias issue.
"""

import pandas as pd
import numpy as np

def analyze_training_data_quarters():
    """Analyze the training data to see if there's a Q1 bias."""
    print("🔍 ANALYZING TRAINING DATA FOR Q1 BIAS")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv('202402_Copy.csv')
    print(f"📊 Loaded {len(df)} rows of training data")
    
    # Add quarter information
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    # Check quarter distribution
    print(f"\n📊 QUARTER DISTRIBUTION:")
    quarter_counts = df['Quarter Num'].value_counts().sort_index()
    for q, count in quarter_counts.items():
        print(f"  Q{q}: {count} records")
    
    # Analyze quarterly growth patterns
    print(f"\n📊 QUARTERLY GROWTH PATTERNS:")
    print("-" * 60)
    
    quarterly_stats = df.groupby('Quarter Num')['ARR YoY Growth (in %)'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    
    print(quarterly_stats)
    
    # Check if Q1 has higher growth rates
    q1_mean = quarterly_stats.loc[1, 'mean']
    q2_mean = quarterly_stats.loc[2, 'mean']
    q3_mean = quarterly_stats.loc[3, 'mean']
    q4_mean = quarterly_stats.loc[4, 'mean']
    
    print(f"\n💡 QUARTERLY GROWTH COMPARISON:")
    print(f"Q1 Mean Growth: {q1_mean:.2f}%")
    print(f"Q2 Mean Growth: {q2_mean:.2f}%")
    print(f"Q3 Mean Growth: {q3_mean:.2f}%")
    print(f"Q4 Mean Growth: {q4_mean:.2f}%")
    
    # Check if Q1 is significantly higher
    if q1_mean > q2_mean and q1_mean > q3_mean and q1_mean > q4_mean:
        print(f"\n🚨 Q1 HAS HIGHEST GROWTH IN TRAINING DATA!")
        print(f"Q1 is {q1_mean - q2_mean:.1f}% higher than Q2")
        print(f"Q1 is {q1_mean - q3_mean:.1f}% higher than Q3")
        print(f"Q1 is {q1_mean - q4_mean:.1f}% higher than Q4")
        print("This explains why Q1 predictions are higher!")
    else:
        print(f"\n🤔 Q1 doesn't have the highest growth in training data...")
        print("The Q1 bias might be coming from somewhere else...")
    
    return df

def analyze_company_examples():
    """Look at specific company examples to understand the pattern."""
    print(f"\n🔍 ANALYZING SPECIFIC COMPANY EXAMPLES")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv('202402_Copy.csv')
    df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
    
    # Get companies with data across all quarters
    companies_with_all_quarters = df.groupby('id_company')['Quarter Num'].nunique()
    complete_companies = companies_with_all_quarters[companies_with_all_quarters >= 4].index[:10]
    
    print(f"📊 Found {len(complete_companies)} companies with data across all quarters")
    print("Looking at first 5 companies:")
    print("-" * 60)
    
    for i, company in enumerate(complete_companies[:5]):
        company_data = df[df['id_company'] == company].sort_values('Quarter Num')
        print(f"\n🏢 {company}:")
        
        q1_growth = None
        q2_growth = None
        q3_growth = None
        q4_growth = None
        
        for _, row in company_data.iterrows():
            quarter = row['Quarter Num']
            growth = row['ARR YoY Growth (in %)']
            arr = row['cARR']
            print(f"  Q{quarter}: ARR=${arr:,.0f}, YoY={growth:.1f}%")
            
            if quarter == 1:
                q1_growth = growth
            elif quarter == 2:
                q2_growth = growth
            elif quarter == 3:
                q3_growth = growth
            elif quarter == 4:
                q4_growth = growth
        
        # Check if Q1 is higher
        if q1_growth and q2_growth and q3_growth and q4_growth:
            if q1_growth > q2_growth and q1_growth > q3_growth and q1_growth > q4_growth:
                print(f"    🚨 Q1 is highest for this company!")
            else:
                print(f"    🤔 Q1 is not highest for this company")

def analyze_model_training_logic():
    """Analyze how the models were trained to understand the bias."""
    print(f"\n🔍 ANALYZING MODEL TRAINING LOGIC")
    print("=" * 60)
    
    print("📊 How the Single-Quarter Models Were Trained:")
    print("-" * 60)
    print("1. Q1 Model: Trained to predict Q1 ARR YoY Growth")
    print("2. Q2 Model: Trained to predict Q2 ARR YoY Growth")
    print("3. Q3 Model: Trained to predict Q3 ARR YoY Growth")
    print("4. Q4 Model: Trained to predict Q4 ARR YoY Growth")
    
    print(f"\n💡 KEY INSIGHT:")
    print("Each model was trained on DIFFERENT target values!")
    print("If Q1 has higher growth rates in the training data,")
    print("then the Q1 model will predict higher values.")
    
    print(f"\n🔍 THE REAL QUESTION:")
    print("Why does Q1 have higher growth rates in the training data?")
    print("Possible reasons:")
    print("1. Q1 is actually a high-growth quarter for SaaS companies")
    print("2. There's a data collection bias")
    print("3. There's a seasonal effect")
    print("4. The YoY calculation is different for Q1")

def analyze_yoy_calculation():
    """Analyze how YoY growth is calculated for different quarters."""
    print(f"\n🔍 ANALYZING YOY CALCULATION LOGIC")
    print("=" * 60)
    
    print("📊 How YoY Growth is Calculated:")
    print("-" * 60)
    print("YoY Growth = (Current Quarter ARR - Same Quarter Previous Year ARR) / Same Quarter Previous Year ARR * 100")
    
    print(f"\n💡 POTENTIAL ISSUE:")
    print("Q1 YoY Growth compares Q1 2024 to Q1 2023")
    print("Q2 YoY Growth compares Q2 2024 to Q2 2023")
    print("Q3 YoY Growth compares Q3 2024 to Q3 2023")
    print("Q4 YoY Growth compares Q4 2024 to Q4 2023")
    
    print(f"\n🤔 POSSIBLE EXPLANATION:")
    print("If companies had lower ARR in Q1 2023 (start of year),")
    print("then Q1 2024 YoY growth would appear higher!")
    print("This could create a systematic Q1 bias in the data.")

def main():
    """Main analysis function."""
    print("🔍 INVESTIGATING Q1 BIAS ISSUE")
    print("=" * 80)
    
    # 1. Analyze training data patterns
    df = analyze_training_data_quarters()
    
    # 2. Look at specific company examples
    analyze_company_examples()
    
    # 3. Analyze model training logic
    analyze_model_training_logic()
    
    # 4. Analyze YoY calculation
    analyze_yoy_calculation()
    
    print(f"\n{'='*80}")
    print("🎯 CONCLUSION")
    print(f"{'='*80}")
    print("The Q1 bias likely comes from:")
    print("1. Q1 having higher growth rates in the training data")
    print("2. Each quarter model being trained on different target distributions")
    print("3. Possible seasonal effects or data collection biases")
    print("4. YoY calculation differences between quarters")
    print("\nThis is NOT a bug - it's a reflection of the training data!")

if __name__ == "__main__":
    main()


