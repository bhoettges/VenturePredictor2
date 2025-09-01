# ==============================================================================
# ENHANCED UNCERTAINTY PREDICTION USING EXISTING MODEL
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.utils import resample
from financial_prediction import load_trained_model, predict_future_arr
from guided_input_system import GuidedInputSystem
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

def predict_with_bootstrap_uncertainty(company_df, n_bootstrap=100, confidence_level=0.8):
    """
    Use bootstrapping on the existing trained model to estimate uncertainty.
    
    Args:
        company_df: Company data for prediction
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (e.g., 0.8 for 80%)
    
    Returns:
        DataFrame with predictions and confidence intervals
    """
    print("🔮 Making predictions with bootstrap uncertainty quantification...")
    
    # Load the existing trained model
    trained_model = load_trained_model('lightgbm_financial_model.pkl')
    
    if trained_model is None:
        print("❌ No trained model found. Please ensure lightgbm_financial_model.pkl exists.")
        return None
    
    # Get the base prediction
    base_forecast = predict_future_arr(trained_model, company_df)
    
    if base_forecast is None or base_forecast.empty:
        print("❌ Base prediction failed.")
        return None
    
    # Extract the model pipeline and feature columns
    model_pipeline = trained_model['model_pipeline']
    feature_cols = trained_model['feature_cols']
    
    # Get the underlying LightGBM estimators
    estimators = model_pipeline.named_steps['model'].estimators_
    
    # Bootstrap predictions
    bootstrap_predictions = []
    
    for i in range(n_bootstrap):
        # Bootstrap the estimators (sample with replacement)
        bootstrap_estimators = np.random.choice(estimators, size=len(estimators), replace=True)
        
        # Create a temporary model pipeline with bootstrapped estimators
        temp_pipeline = model_pipeline
        temp_pipeline.named_steps['model'].estimators_ = bootstrap_estimators
        
        # Make prediction with bootstrapped model
        try:
            temp_forecast = predict_future_arr({'model_pipeline': temp_pipeline, 'feature_cols': feature_cols}, company_df)
            if temp_forecast is not None and not temp_forecast.empty:
                bootstrap_predictions.append(temp_forecast['Predicted YoY Growth (%)'].values)
        except:
            continue
    
    if len(bootstrap_predictions) < 10:
        print("❌ Not enough successful bootstrap predictions.")
        return base_forecast
    
    # Convert to numpy array
    bootstrap_array = np.array(bootstrap_predictions)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Calculate percentiles for each quarter
    lower_bounds = np.percentile(bootstrap_array, lower_percentile, axis=0)
    upper_bounds = np.percentile(bootstrap_array, upper_percentile, axis=0)
    median_predictions = np.percentile(bootstrap_array, 50, axis=0)
    
    # Create enhanced results
    enhanced_results = base_forecast.copy()
    enhanced_results['Pessimistic'] = lower_bounds
    enhanced_results['Realistic'] = median_predictions
    enhanced_results['Optimistic'] = upper_bounds
    enhanced_results['Uncertainty_Range'] = upper_bounds - lower_bounds
    
    # Calculate absolute ARR values
    last_known_arr = company_df.iloc[-1]['cARR'] if 'cARR' in company_df.columns else 2100000
    
    enhanced_results['ARR_Pessimistic'] = last_known_arr * (1 + lower_bounds / 100)
    enhanced_results['ARR_Realistic'] = last_known_arr * (1 + median_predictions / 100)
    enhanced_results['ARR_Optimistic'] = last_known_arr * (1 + upper_bounds / 100)
    
    return enhanced_results

def predict_with_ensemble_uncertainty(company_df, n_estimators=10):
    """
    Use ensemble of predictions from the existing model to estimate uncertainty.
    """
    print("🔮 Making predictions with ensemble uncertainty quantification...")
    
    # Load the existing trained model
    trained_model = load_trained_model('lightgbm_financial_model.pkl')
    
    if trained_model is None:
        print("❌ No trained model found.")
        return None
    
    # Get the model pipeline
    model_pipeline = trained_model['model_pipeline']
    feature_cols = trained_model['feature_cols']
    
    # Get the underlying LightGBM estimators
    estimators = model_pipeline.named_steps['model'].estimators_
    
    # Make predictions with different subsets of estimators
    ensemble_predictions = []
    
    for i in range(n_estimators):
        # Randomly sample estimators
        sample_size = max(1, len(estimators) // 2)
        sampled_estimators = np.random.choice(estimators, size=sample_size, replace=False)
        
        # Create temporary model
        temp_pipeline = model_pipeline
        temp_pipeline.named_steps['model'].estimators_ = sampled_estimators
        
        # Make prediction
        try:
            temp_forecast = predict_future_arr({'model_pipeline': temp_pipeline, 'feature_cols': feature_cols}, company_df)
            if temp_forecast is not None and not temp_forecast.empty:
                ensemble_predictions.append(temp_forecast['Predicted YoY Growth (%)'].values)
        except:
            continue
    
    if len(ensemble_predictions) < 5:
        print("❌ Not enough successful ensemble predictions.")
        return predict_future_arr(trained_model, company_df)
    
    # Calculate statistics
    ensemble_array = np.array(ensemble_predictions)
    
    # Calculate percentiles
    pessimistic = np.percentile(ensemble_array, 10, axis=0)
    realistic = np.percentile(ensemble_array, 50, axis=0)
    optimistic = np.percentile(ensemble_array, 90, axis=0)
    
    # Get base prediction for structure
    base_forecast = predict_future_arr(trained_model, company_df)
    
    # Create enhanced results
    enhanced_results = base_forecast.copy()
    enhanced_results['Pessimistic'] = pessimistic
    enhanced_results['Realistic'] = realistic
    enhanced_results['Optimistic'] = optimistic
    enhanced_results['Uncertainty_Range'] = optimistic - pessimistic
    
    # Calculate absolute ARR values
    last_known_arr = company_df.iloc[-1]['cARR'] if 'cARR' in company_df.columns else 2100000
    
    enhanced_results['ARR_Pessimistic'] = last_known_arr * (1 + pessimistic / 100)
    enhanced_results['ARR_Realistic'] = last_known_arr * (1 + realistic / 100)
    enhanced_results['ARR_Optimistic'] = last_known_arr * (1 + optimistic / 100)
    
    return enhanced_results

def visualize_uncertainty_forecast(results_df, title="ARR Growth Forecast with Uncertainty"):
    """Create a visualization of the uncertainty forecast."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Growth Rate Uncertainty
    quarters = results_df['Future Quarter']
    pessimistic = results_df['Pessimistic']
    realistic = results_df['Realistic']
    optimistic = results_df['Optimistic']
    
    ax1.fill_between(quarters, pessimistic, optimistic, alpha=0.3, label='Uncertainty Range (10th-90th %)')
    ax1.plot(quarters, realistic, 'o-', linewidth=2, markersize=8, label='Realistic (50th %)')
    ax1.plot(quarters, pessimistic, 's--', alpha=0.7, label='Pessimistic (10th %)')
    ax1.plot(quarters, optimistic, '^--', alpha=0.7, label='Optimistic (90th %)')
    
    ax1.set_title(f'{title} - Growth Rates', fontsize=16, fontweight='bold')
    ax1.set_ylabel('YoY Growth Rate (%)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute ARR Uncertainty
    arr_pessimistic = results_df['ARR_Pessimistic']
    arr_realistic = results_df['ARR_Realistic']
    arr_optimistic = results_df['ARR_Optimistic']
    
    ax2.fill_between(quarters, arr_pessimistic, arr_optimistic, alpha=0.3, label='ARR Uncertainty Range')
    ax2.plot(quarters, arr_realistic, 'o-', linewidth=2, markersize=8, label='Expected ARR (50th %)')
    ax2.plot(quarters, arr_pessimistic, 's--', alpha=0.7, label='Pessimistic ARR (10th %)')
    ax2.plot(quarters, arr_optimistic, '^--', alpha=0.7, label='Optimistic ARR (90th %)')
    
    ax2.set_title(f'{title} - Absolute ARR', fontsize=16, fontweight='bold')
    ax2.set_ylabel('ARR (€)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_uncertainty_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_enhanced_uncertainty():
    """Test the enhanced uncertainty prediction with your existing model."""
    print("🎯 TESTING ENHANCED UNCERTAINTY PREDICTION")
    print("=" * 60)
    
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
    
    # Test ensemble uncertainty prediction
    print("\n🔮 Testing Ensemble Uncertainty Prediction...")
    results = predict_with_ensemble_uncertainty(forecast_df, n_estimators=20)
    
    if results is not None:
        # Display results
        print(f"\n🔮 ENHANCED UNCERTAINTY FORECAST FOR 2025:")
        print("=" * 60)
        
        for _, row in results.iterrows():
            print(f"\n{row['Future Quarter']}:")
            print(f"  Pessimistic (10th %): {row['Pessimistic']:.1f}% growth")
            print(f"  Realistic (50th %):   {row['Realistic']:.1f}% growth")
            print(f"  Optimistic (90th %):  {row['Optimistic']:.1f}% growth")
            print(f"  Uncertainty Range:    ±{row['Uncertainty_Range']/2:.1f}%")
            print(f"  Expected ARR:         €{row['ARR_Realistic']:,.0f}")
        
        # Create visualization
        visualize_uncertainty_forecast(results, "Enhanced Uncertainty Forecast")
        
        return results
    else:
        print("❌ Enhanced uncertainty prediction failed.")
        return None

if __name__ == "__main__":
    results = test_enhanced_uncertainty()
