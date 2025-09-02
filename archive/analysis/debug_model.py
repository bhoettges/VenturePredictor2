#!/usr/bin/env python3
"""Debug script to analyze the model prediction issue."""

import pandas as pd
import numpy as np

def analyze_training_data():
    """Analyze the training data to understand the growth rate distribution."""
    print("🔍 ANALYZING TRAINING DATA")
    print("=" * 50)
    
    df = pd.read_csv('202402_Copy.csv')
    
    print("📊 ARR YoY Growth Statistics:")
    print(df['ARR YoY Growth (in %)'].describe())
    
    print("\n📈 Growth Rate Distribution:")
    growth = df['ARR YoY Growth (in %)'].dropna()
    
    # Check for extreme values
    print(f"Values > 10 (1000% growth): {len(growth[growth > 10])}")
    print(f"Values > 5 (500% growth): {len(growth[growth > 5])}")
    print(f"Values > 2 (200% growth): {len(growth[growth > 2])}")
    print(f"Values > 1 (100% growth): {len(growth[growth > 1])}")
    
    # Show some extreme examples
    print("\n🚨 Extreme Growth Examples:")
    extreme = growth[growth > 5].head(10)
    for i, val in enumerate(extreme):
        print(f"  {i+1}. {val:.2f} ({val*100:.0f}% growth)")
    
    # Check reasonable growth rates
    print("\n✅ Reasonable Growth Examples (0-100%):")
    reasonable = growth[(growth >= 0) & (growth <= 1)].head(10)
    for i, val in enumerate(reasonable):
        print(f"  {i+1}. {val:.3f} ({val*100:.1f}% growth)")

def analyze_test_data():
    """Analyze the test company data."""
    print("\n🔍 ANALYZING TEST COMPANY DATA")
    print("=" * 50)
    
    df = pd.read_csv('test_company_2024.csv')
    print("Test company data:")
    print(df)
    
    latest = df.iloc[-1]
    growth_rate = (latest['Quarterly_Net_New_ARR'] / latest['ARR_End_of_Quarter']) * 100
    
    print(f"\n📊 Latest Quarter Analysis:")
    print(f"ARR: ${latest['ARR_End_of_Quarter']:,.0f}")
    print(f"Net New ARR: ${latest['Quarterly_Net_New_ARR']:,.0f}")
    print(f"Calculated Growth Rate: {growth_rate:.1f}%")
    
    # Check if this growth rate is reasonable
    if growth_rate > 100:
        print("⚠️  WARNING: Growth rate > 100% is very high!")
    elif growth_rate > 50:
        print("⚠️  WARNING: Growth rate > 50% is quite high!")
    else:
        print("✅ Growth rate looks reasonable")

def suggest_fixes():
    """Suggest fixes for the unrealistic predictions."""
    print("\n🔧 SUGGESTED FIXES")
    print("=" * 50)
    
    print("1. 🧹 CLEAN TRAINING DATA:")
    print("   - Remove extreme outliers (>500% growth)")
    print("   - Cap growth rates at reasonable levels (e.g., 200%)")
    print("   - Focus on companies with realistic growth patterns")
    
    print("\n2. 🎯 RETRAIN MODEL:")
    print("   - Use cleaned data with realistic growth rates")
    print("   - Consider using log-transformed growth rates")
    print("   - Add constraints to prevent extreme predictions")
    
    print("\n3. 🔄 ADJUST PREDICTION LOGIC:")
    print("   - Cap predictions at reasonable levels")
    print("   - Use industry benchmarks for validation")
    print("   - Implement sanity checks on output")

if __name__ == "__main__":
    analyze_training_data()
    analyze_test_data()
    suggest_fixes()
