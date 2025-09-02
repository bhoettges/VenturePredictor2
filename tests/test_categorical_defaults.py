#!/usr/bin/env python3
"""
Test script to demonstrate the difference between categorical default approaches
"""

import pandas as pd
import numpy as np

def analyze_training_data_categoricals():
    """Analyze categorical features in training data."""
    print("🔍 ANALYZING CATEGORICAL FEATURES IN TRAINING DATA")
    print("=" * 60)
    
    try:
        # Load training data
        df = pd.read_csv('202402_Copy.csv')
        print(f"✅ Loaded {len(df)} rows of training data")
        
        # Analyze categorical features
        categorical_features = ['Currency', 'Sector', 'Target Customer', 'Country', 'Deal Team']
        
        for feature in categorical_features:
            if feature in df.columns:
                value_counts = df[feature].value_counts()
                most_common = value_counts.index[0]
                most_common_count = value_counts.iloc[0]
                total_count = len(df)
                percentage = (most_common_count / total_count) * 100
                
                print(f"\n📊 {feature}:")
                print(f"  Most common value: {most_common} ({most_common_count:,} times, {percentage:.1f}%)")
                print(f"  Unique values: {len(value_counts)}")
                print(f"  Top 5 values: {value_counts.head().to_dict()}")
            else:
                print(f"\n❌ {feature}: Not found in training data")
        
        # Show distribution summary
        print(f"\n📈 CATEGORICAL DISTRIBUTION SUMMARY:")
        print("-" * 40)
        for feature in categorical_features:
            if feature in df.columns:
                mode_value = df[feature].mode().iloc[0]
                print(f"{feature}: Mode = {mode_value}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def test_different_approaches():
    """Test different approaches for categorical defaults."""
    print("\n🧪 TESTING DIFFERENT CATEGORICAL DEFAULT APPROACHES")
    print("=" * 60)
    
    # Simulate user input
    user_arr = 2800000
    user_growth = 180
    
    print(f"📊 User Input:")
    print(f"  ARR: ${user_arr:,.0f}")
    print(f"  Growth: {user_growth:.1f}%")
    
    # Approach 1: Arbitrary assumptions (old way)
    print(f"\n❌ APPROACH 1: ARBITRARY ASSUMPTIONS (OLD)")
    print("-" * 40)
    arbitrary_defaults = {
        'Sector': 1,  # Assumes "Software"
        'Country': 1,  # Assumes "United States"
        'Currency': 1,  # Assumes "USD"
        'Target Customer': 1,  # Assumes "Enterprise"
        'Deal Team': 1  # Assumes "Sales & Productivity"
    }
    for key, value in arbitrary_defaults.items():
        print(f"  {key}: {value} (arbitrary assumption)")
    
    # Approach 2: Use 0 (neutral/unknown)
    print(f"\n⚠️ APPROACH 2: USE 0 (NEUTRAL/UNKNOWN)")
    print("-" * 40)
    neutral_defaults = {
        'Sector': 0,
        'Country': 0,
        'Currency': 0,
        'Target Customer': 0,
        'Deal Team': 0
    }
    for key, value in neutral_defaults.items():
        print(f"  {key}: {value} (neutral/unknown)")
    
    # Approach 3: Data-driven (new way)
    print(f"\n✅ APPROACH 3: DATA-DRIVEN (NEW)")
    print("-" * 40)
    df = analyze_training_data_categoricals()
    if df is not None:
        data_driven_defaults = {}
        for feature in ['Sector', 'Country', 'Currency', 'Target Customer', 'Deal Team']:
            if feature in df.columns:
                mode_value = df[feature].mode().iloc[0]
                data_driven_defaults[feature] = mode_value
                print(f"  {feature}: {mode_value} (most common in training data)")
            else:
                data_driven_defaults[feature] = 0
                print(f"  {feature}: 0 (not found in training data)")
    
    return arbitrary_defaults, neutral_defaults, data_driven_defaults

def compare_impact_on_predictions():
    """Compare how different categorical approaches might impact predictions."""
    print(f"\n🎯 IMPACT ON PREDICTIONS")
    print("=" * 60)
    
    print("📊 How categorical defaults affect model predictions:")
    print("\n❌ Arbitrary Assumptions (Sector=1, Country=1):")
    print("  • Forces specific sector/country characteristics")
    print("  • May introduce bias if training data has different distribution")
    print("  • Assumes user is in most common category")
    
    print("\n⚠️ Neutral Values (Sector=0, Country=0):")
    print("  • Treats as unknown/neutral category")
    print("  • Model uses baseline characteristics")
    print("  • Less biased but may lose sector-specific patterns")
    
    print("\n✅ Data-Driven (Most Common Values):")
    print("  • Uses actual training data distribution")
    print("  • Reflects real-world patterns")
    print("  • Most representative of typical companies")
    
    print(f"\n💡 RECOMMENDATION:")
    print("  Use data-driven approach (most common values) for best results!")

def main():
    """Run the categorical analysis."""
    print("🚀 CATEGORICAL DEFAULTS ANALYSIS")
    print("=" * 70)
    
    # Analyze training data
    df = analyze_training_data_categoricals()
    
    # Test different approaches
    arbitrary, neutral, data_driven = test_different_approaches()
    
    # Compare impact
    compare_impact_on_predictions()
    
    print(f"\n✅ Analysis complete!")
    print(f"\n📋 Key Takeaways:")
    print(f"  • Arbitrary assumptions (1) can introduce bias")
    print(f"  • Neutral values (0) are safer but less informative")
    print(f"  • Data-driven defaults (most common) are most representative")
    print(f"  • Always use training data distribution when possible")

if __name__ == "__main__":
    main()
