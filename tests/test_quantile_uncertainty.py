#!/usr/bin/env python3
"""
Test our quantile regression uncertainty quantification with real 2024 data.
"""

from quantile_regression import load_quantile_models, predict_with_uncertainty, visualize_uncertainty_forecast
from guided_input_system import GuidedInputSystem

def test_uncertainty_forecast():
    """
    Test the quantile regression approach with uncertainty quantification.
    """
    print("🎯 TESTING QUANTILE REGRESSION WITH UNCERTAINTY")
    print("=" * 60)
    
    # Load quantile models
    quantile_model_data = load_quantile_models('quantile_models.pkl')
    
    if quantile_model_data is None:
        print("❌ Quantile models not found. Please train them first.")
        print("Run: python3.10 quantile_regression.py")
        return
    
    # Create test data using guided input system
    guided_system = GuidedInputSystem()
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
    
    # Make predictions with uncertainty
    results = predict_with_uncertainty(quantile_model_data, forecast_df)
    
    # Display results
    print(f"\n🔮 UNCERTAINTY FORECAST FOR 2025:")
    print("=" * 60)
    
    for _, row in results.iterrows():
        print(f"\n{row['Future Quarter']}:")
        print(f"  Pessimistic (10th %): {row['Pessimistic (10th %)']:.1f}% growth")
        print(f"  Realistic (50th %):   {row['Realistic (50th %)']:.1f}% growth")
        print(f"  Optimistic (90th %):  {row['Optimistic (90th %)']:.1f}% growth")
        print(f"  Uncertainty Range:    ±{row['Uncertainty Range']/2:.1f}%")
        print(f"  Expected ARR:         €{row['ARR_50th']:,.0f}")
    
    # Create visualization
    visualize_uncertainty_forecast(results)
    
    return results

if __name__ == "__main__":
    results = test_uncertainty_forecast()
