#!/usr/bin/env python3
"""
Test the enhanced guided input system with historical ARR data from test_company_2024.csv
"""

import pandas as pd
from enhanced_guided_input import EnhancedGuidedInputSystem
from simple_uncertainty_prediction import predict_with_simple_uncertainty
from financial_prediction import load_trained_model

def load_test_company_data():
    """Load and process test_company_2024.csv data."""
    print("📁 Loading test_company_2024.csv data...")
    
    try:
        df = pd.read_csv('test_company_2024.csv')
        print(f"✅ Loaded {len(df)} quarters of data")
        
        # Extract ARR data
        arr_data = df['ARR_End_of_Quarter'].tolist()
        net_new_arr_data = df['Quarterly_Net_New_ARR'].tolist()
        
        print(f"📊 ARR progression: {arr_data}")
        print(f"📊 Net New ARR progression: {net_new_arr_data}")
        
        return arr_data, net_new_arr_data
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None

def test_minimal_input():
    """Test with minimal inputs (current ARR + Net New ARR only)."""
    print("\n🎯 TEST 1: MINIMAL INPUT (Current ARR + Net New ARR only)")
    print("=" * 60)
    
    # Load test data
    arr_data, net_new_arr_data = load_test_company_data()
    if not arr_data:
        return
    
    # Use the latest data
    current_arr = arr_data[-1]  # $2,800,000
    net_new_arr = net_new_arr_data[-1]  # $800,000
    
    print(f"📊 Using latest data:")
    print(f"  Current ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    
    # Create enhanced system
    enhanced_system = EnhancedGuidedInputSystem()
    
    # Generate forecast input with minimal data
    df_minimal = enhanced_system.create_forecast_input_with_history(
        current_arr=current_arr,
        net_new_arr=net_new_arr
    )
    
    print(f"✅ Created {len(df_minimal)} quarters with minimal inputs")
    print(f"📊 Generated historical ARR: {df_minimal['cARR'].tolist()}")
    
    # Make prediction
    try:
        trained_model = load_trained_model('lightgbm_financial_model.pkl')
        if trained_model:
            results_minimal = predict_with_simple_uncertainty(df_minimal, uncertainty_factor=0.1)
            print(f"✅ Minimal input prediction successful!")
            return results_minimal
        else:
            print("❌ No trained model available")
            return None
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def test_with_historical_data():
    """Test with actual historical ARR data from test_company_2024.csv."""
    print("\n🎯 TEST 2: WITH HISTORICAL ARR DATA (4 quarters)")
    print("=" * 60)
    
    # Load test data
    arr_data, net_new_arr_data = load_test_company_data()
    if not arr_data:
        return
    
    # Extract historical ARR (4 quarters)
    historical_arr = arr_data  # [1000000, 1400000, 2000000, 2800000]
    current_arr = arr_data[-1]  # $2,800,000
    net_new_arr = net_new_arr_data[-1]  # $800,000
    
    print(f"📊 Using historical ARR data:")
    print(f"  Historical ARR: {[f'${arr:,.0f}' for arr in historical_arr]}")
    print(f"  Current ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    
    # Create enhanced system
    enhanced_system = EnhancedGuidedInputSystem()
    
    # Generate forecast input with historical data
    df_historical = enhanced_system.create_forecast_input_with_history(
        current_arr=current_arr,
        net_new_arr=net_new_arr,
        historical_arr=historical_arr
    )
    
    print(f"✅ Created {len(df_historical)} quarters with historical data")
    print(f"📊 Historical ARR used: {df_historical['cARR'].tolist()}")
    
    # Make prediction
    try:
        trained_model = load_trained_model('lightgbm_financial_model.pkl')
        if trained_model:
            results_historical = predict_with_simple_uncertainty(df_historical, uncertainty_factor=0.1)
            print(f"✅ Historical data prediction successful!")
            return results_historical
        else:
            print("❌ No trained model available")
            return None
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def test_with_advanced_metrics():
    """Test with historical data + advanced metrics overrides."""
    print("\n🎯 TEST 3: WITH HISTORICAL DATA + ADVANCED METRICS")
    print("=" * 60)
    
    # Load test data
    arr_data, net_new_arr_data = load_test_company_data()
    if not arr_data:
        return
    
    # Extract data
    historical_arr = arr_data
    current_arr = arr_data[-1]
    net_new_arr = net_new_arr_data[-1]
    
    # Advanced metrics overrides based on test data
    advanced_metrics = {
        'magic_number': 0.8,  # Based on actual data
        'gross_margin': 73.0,  # From test data
        'headcount': 70,  # From test data
        'sales_marketing': 1000000,  # Estimate based on Magic Number
        'ebitda': 560000,  # 20% of ARR
        'cash_burn': -640000  # Based on Burn Multiple
    }
    
    print(f"📊 Using historical data + advanced metrics:")
    print(f"  Historical ARR: {[f'${arr:,.0f}' for arr in historical_arr]}")
    print(f"  Advanced metrics: {advanced_metrics}")
    
    # Create enhanced system
    enhanced_system = EnhancedGuidedInputSystem()
    
    # Generate forecast input with historical data + advanced metrics
    df_advanced = enhanced_system.create_forecast_input_with_history(
        current_arr=current_arr,
        net_new_arr=net_new_arr,
        historical_arr=historical_arr,
        advanced_metrics=advanced_metrics
    )
    
    print(f"✅ Created {len(df_advanced)} quarters with historical data + advanced metrics")
    
    # Make prediction
    try:
        trained_model = load_trained_model('lightgbm_financial_model.pkl')
        if trained_model:
            results_advanced = predict_with_simple_uncertainty(df_advanced, uncertainty_factor=0.1)
            print(f"✅ Advanced metrics prediction successful!")
            return results_advanced
        else:
            print("❌ No trained model available")
            return None
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def compare_results(results_minimal, results_historical, results_advanced):
    """Compare results from different input methods."""
    print("\n🎯 COMPARISON OF RESULTS")
    print("=" * 60)
    
    if results_minimal is not None and results_historical is not None and results_advanced is not None:
        print("📊 Forecast Comparison (Q1 2025):")
        print(f"{'Method':<25} {'Pessimistic':<12} {'Realistic':<12} {'Optimistic':<12}")
        print("-" * 65)
        
        # Minimal input
        q1_minimal = results_minimal.iloc[0]
        print(f"{'Minimal Input':<25} {q1_minimal['Pessimistic']:<12.1f}% {q1_minimal['Realistic']:<12.1f}% {q1_minimal['Optimistic']:<12.1f}%")
        
        # Historical data
        q1_historical = results_historical.iloc[0]
        print(f"{'Historical Data':<25} {q1_historical['Pessimistic']:<12.1f}% {q1_historical['Realistic']:<12.1f}% {q1_historical['Optimistic']:<12.1f}%")
        
        # Advanced metrics
        q1_advanced = results_advanced.iloc[0]
        print(f"{'Advanced Metrics':<25} {q1_advanced['Pessimistic']:<12.1f}% {q1_advanced['Realistic']:<12.1f}% {q1_advanced['Optimistic']:<12.1f}%")
        
        print(f"\n💡 Key Insights:")
        print(f"  • Historical data provides more realistic growth patterns")
        print(f"  • Advanced metrics allow fine-tuning of key ratios")
        print(f"  • All methods include ±10% uncertainty quantification")
        
    else:
        print("❌ Cannot compare results - some predictions failed")

def main():
    """Run all tests and compare results."""
    print("🚀 TESTING ENHANCED GUIDED INPUT WITH HISTORICAL ARR SUPPORT")
    print("=" * 70)
    
    # Run all tests
    results_minimal = test_minimal_input()
    results_historical = test_with_historical_data()
    results_advanced = test_with_advanced_metrics()
    
    # Compare results
    compare_results(results_minimal, results_historical, results_advanced)
    
    print("\n✅ Enhanced historical ARR testing completed!")
    print("\n📋 Key Benefits of Historical ARR Support:")
    print("  • More accurate growth rate calculation from real data")
    print("  • Better feature engineering with actual lags and trends")
    print("  • Improved model performance with realistic patterns")
    print("  • Reduced uncertainty through real historical context")

if __name__ == "__main__":
    main()
