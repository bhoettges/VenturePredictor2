#!/usr/bin/env python3
"""
Test our financial forecasting model with real 2024 data.
"""

from enhanced_prediction import EnhancedFinancialPredictor

def test_with_real_data():
    """Test the model with the user's real 2024 Q4 data."""
    
    print("🎯 TESTING WITH REAL 2024 DATA")
    print("=" * 50)
    
    # Extract key metrics from Q4 2024 data
    current_arr = 2100000  # ARR_End_of_Quarter Q4 2024
    net_new_arr = 320000   # Quarterly_Net_New_ARR Q4 2024
    growth_rate = (net_new_arr / (current_arr - net_new_arr)) * 100  # Calculate YoY growth
    
    print(f"📊 REAL DATA FROM Q4 2024:")
    print(f"  Current ARR: ${current_arr:,.0f}")
    print(f"  Net New ARR: ${net_new_arr:,.0f}")
    print(f"  Calculated Growth Rate: {growth_rate:.1f}%")
    print(f"  Headcount: 38 employees")
    print(f"  Gross Margin: 82%")
    
    # Initialize the enhanced predictor
    predictor = EnhancedFinancialPredictor()
    
    # Create input data manually
    from guided_input_system import GuidedInputSystem
    guided_system = GuidedInputSystem()
    guided_system.initialize_from_training_data()
    
    # Build primary inputs
    primary_inputs = {
        'cARR': current_arr,
        'Net New ARR': net_new_arr,
        'ARR YoY Growth (in %)': growth_rate,
        'Quarter Num': 1
    }
    
    # Infer secondary metrics
    inferred_metrics = guided_system.infer_secondary_metrics(primary_inputs)
    
    # Override with actual known values
    inferred_metrics['Headcount (HC)'] = 38  # Actual headcount
    inferred_metrics['Gross Margin (in %)'] = 82  # Actual gross margin
    inferred_metrics['id_company'] = 'Test Company 2024'
    inferred_metrics['Financial Quarter'] = 'FY24 Q4'
    
    # Create forecast-ready DataFrame
    forecast_df = guided_system.create_forecast_input(inferred_metrics)
    
    # Try to make prediction with trained model
    try:
        from financial_prediction import load_trained_model, predict_future_arr
        trained_model = load_trained_model('lightgbm_financial_model.pkl')
        if trained_model:
            forecast_results = predict_future_arr(trained_model, forecast_df)
            model_used = "Trained Model"
            forecast_success = True
        else:
            raise Exception("No trained model available")
    except Exception as e:
        # Use fallback calculation
        forecast_results = predictor._generate_fallback_forecast(inferred_metrics)
        model_used = "Fallback Calculation"
        forecast_success = False
    
    # Generate insights
    insights = predictor._generate_insights(inferred_metrics, forecast_results)
    
    # Display results
    print(f"\n🔮 FORECAST FOR 2025:")
    print("=" * 50)
    
    if not forecast_results.empty:
        print(f"{'Quarter':<12} {'YoY Growth':<12} {'Absolute ARR':<15}")
        print("-" * 50)
        
        for _, row in forecast_results.iterrows():
            quarter = row['Future Quarter']
            growth = row['Predicted YoY Growth (%)']
            arr = row['Predicted Absolute cARR (€)']
            print(f"{quarter:<12} {growth:>8.1f}%    ${arr:>12,.0f}")
    
    print(f"\n💡 INSIGHTS:")
    print(f"  Company Stage: {insights['size_category']}")
    print(f"  {insights['size_insight']}")
    print(f"  Growth Analysis: {insights['growth_insight']}")
    print(f"  Efficiency: {insights['efficiency_insight']}")
    
    if 'forecast_insight' in insights:
        print(f"  Forecast Trend: {insights['forecast_insight']}")
    
    print(f"\n🔧 TECHNICAL DETAILS:")
    print(f"  Model Used: {model_used}")
    print(f"  Forecast Success: {'✅ Yes' if forecast_success else '⚠️ Fallback'}")
    
    return forecast_results

if __name__ == "__main__":
    results = test_with_real_data() 