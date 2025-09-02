#!/usr/bin/env python3
"""
Simple uncertainty prediction using the existing trained model.
"""

from financial_prediction import load_trained_model, predict_future_arr
from enhanced_guided_input import EnhancedGuidedInputSystem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def predict_with_simple_uncertainty(company_df, uncertainty_factor=0.1):
    """
    Make predictions with simple uncertainty quantification using the existing model.
    
    Args:
        company_df: Company data for prediction
        uncertainty_factor: Factor to determine uncertainty range (default 0.1 = ±10%)
    
    Returns:
        DataFrame with predictions and uncertainty ranges
    """
    print("🔮 Making predictions with simple uncertainty quantification...")
    
    # Load the existing trained model
    trained_model = load_trained_model('lightgbm_financial_model.pkl')
    
    if trained_model is None:
        print("❌ No trained model found.")
        return None
    
    # Get the base prediction
    base_forecast = predict_future_arr(trained_model, company_df)
    
    if base_forecast is None or base_forecast.empty:
        print("❌ Base prediction failed.")
        return None
    
    # Add uncertainty ranges
    enhanced_results = base_forecast.copy()
    
    # Calculate uncertainty ranges based on the predicted growth rates
    for i, row in enhanced_results.iterrows():
        predicted_growth = row['Predicted YoY Growth (%)']
        
        # Calculate uncertainty range
        uncertainty_range = predicted_growth * uncertainty_factor
        
        # Add uncertainty columns
        enhanced_results.loc[i, 'Pessimistic'] = max(0, predicted_growth - uncertainty_range)
        enhanced_results.loc[i, 'Realistic'] = predicted_growth
        enhanced_results.loc[i, 'Optimistic'] = predicted_growth + uncertainty_range
        enhanced_results.loc[i, 'Uncertainty_Range'] = uncertainty_range * 2
        
        # Calculate absolute ARR values
        last_known_arr = company_df.iloc[-1]['cARR'] if 'cARR' in company_df.columns else 2100000
        
        enhanced_results.loc[i, 'ARR_Pessimistic'] = last_known_arr * (1 + enhanced_results.loc[i, 'Pessimistic'] / 100)
        enhanced_results.loc[i, 'ARR_Realistic'] = last_known_arr * (1 + enhanced_results.loc[i, 'Realistic'] / 100)
        enhanced_results.loc[i, 'ARR_Optimistic'] = last_known_arr * (1 + enhanced_results.loc[i, 'Optimistic'] / 100)
    
    return enhanced_results

def visualize_uncertainty_forecast(results_df, title="ARR Growth Forecast with Uncertainty"):
    """Create a visualization of the uncertainty forecast."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Growth Rate Uncertainty
    quarters = results_df['Future Quarter']
    pessimistic = results_df['Pessimistic']
    realistic = results_df['Realistic']
    optimistic = results_df['Optimistic']
    
    ax1.fill_between(quarters, pessimistic, optimistic, alpha=0.3, label='Uncertainty Range (±10%)')
    ax1.plot(quarters, realistic, 'o-', linewidth=2, markersize=8, label='Realistic (Model Prediction)')
    ax1.plot(quarters, pessimistic, 's--', alpha=0.7, label='Pessimistic')
    ax1.plot(quarters, optimistic, '^--', alpha=0.7, label='Optimistic')
    
    ax1.set_title(f'{title} - Growth Rates', fontsize=16, fontweight='bold')
    ax1.set_ylabel('YoY Growth Rate (%)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute ARR Uncertainty
    arr_pessimistic = results_df['ARR_Pessimistic']
    arr_realistic = results_df['ARR_Realistic']
    arr_optimistic = results_df['ARR_Optimistic']
    
    ax2.fill_between(quarters, arr_pessimistic, arr_optimistic, alpha=0.3, label='ARR Uncertainty Range')
    ax2.plot(quarters, arr_realistic, 'o-', linewidth=2, markersize=8, label='Expected ARR')
    ax2.plot(quarters, arr_pessimistic, 's--', alpha=0.7, label='Pessimistic ARR')
    ax2.plot(quarters, arr_optimistic, '^--', alpha=0.7, label='Optimistic ARR')
    
    ax2.set_title(f'{title} - Absolute ARR', fontsize=16, fontweight='bold')
    ax2.set_ylabel('ARR (€)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_uncertainty_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_simple_uncertainty():
    """Test the simple uncertainty prediction with your existing model."""
    print("🎯 TESTING SIMPLE UNCERTAINTY PREDICTION")
    print("=" * 60)
    
    # Create test data using guided input system
    guided_system = EnhancedGuidedInputSystem()
    guided_system.initialize_from_training_data()
    
    # Build primary inputs (same as before)
    primary_inputs = {
        'cARR': 2100000,  # $2.1M ARR
        'Net New ARR': 320000,  # $320K net new
        'ARR YoY Growth (in %)': 18.0,  # 18% growth
        'Quarter Num': 1
    }
    
    # Infer secondary metrics
    inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
    inferred_metrics['Headcount (HC)'] = 38
    inferred_metrics['Gross Margin (in %)'] = 82
    inferred_metrics['id_company'] = 'Test Company 2024'
    inferred_metrics['Financial Quarter'] = 'FY24 Q4'
    
    # Create forecast-ready DataFrame
    forecast_df = guided_system.create_forecast_input(inferred_metrics)
    
    # Test simple uncertainty prediction
    print("\n🔮 Testing Simple Uncertainty Prediction...")
    results = predict_with_simple_uncertainty(forecast_df, uncertainty_factor=0.1)
    
    if results is not None:
        # Display results
        print(f"\n🔮 SIMPLE UNCERTAINTY FORECAST FOR 2025:")
        print("=" * 60)
        
        for _, row in results.iterrows():
            print(f"\n{row['Future Quarter']}:")
            print(f"  Pessimistic:        {row['Pessimistic']:.1f}% growth")
            print(f"  Realistic:          {row['Realistic']:.1f}% growth")
            print(f"  Optimistic:         {row['Optimistic']:.1f}% growth")
            print(f"  Uncertainty Range:  ±{row['Uncertainty_Range']/2:.1f}%")
            print(f"  Expected ARR:       €{row['ARR_Realistic']:,.0f}")
        
        # Create visualization
        visualize_uncertainty_forecast(results, "Simple Uncertainty Forecast")
        
        return results
    else:
        print("❌ Simple uncertainty prediction failed.")
        return None

if __name__ == "__main__":
    results = test_simple_uncertainty()
