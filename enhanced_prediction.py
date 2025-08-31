import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from guided_input_system import GuidedInputSystem
from financial_prediction import load_trained_model, predict_future_arr

class EnhancedFinancialPredictor:
    """
    Enhanced financial prediction system that combines guided input with intelligent forecasting.
    Provides a complete user experience from input to forecast.
    """
    
    def __init__(self, model_path: str = 'lightgbm_financial_model.pkl'):
        """
        Initialize the enhanced predictor.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.guided_system = GuidedInputSystem()
        self.trained_model = None
        self.is_ready = False
        
    def initialize(self):
        """Initialize the system by loading the model and guided input system."""
        print("🚀 Initializing Enhanced Financial Predictor...")
        
        try:
            # Load trained model
            self.trained_model = load_trained_model(self.model_path)
            if self.trained_model is None:
                print("⚠️ Warning: Could not load trained model. Using guided input only.")
            
            # Initialize guided input system
            self.guided_system.initialize_from_training_data()
            
            self.is_ready = True
            print("✅ Enhanced Financial Predictor ready!")
            
        except Exception as e:
            print(f"❌ Error initializing: {e}")
            self.is_ready = False
    
    def run_guided_forecast(self) -> Dict:
        """
        Run the complete guided forecasting workflow.
        
        Returns:
            Dictionary containing forecast results and metadata
        """
        if not self.is_ready:
            self.initialize()
        
        print("\n" + "="*60)
        print("🎯 ENHANCED FINANCIAL FORECASTING SYSTEM")
        print("="*60)
        
        # Step 1: Get user inputs through guided system
        print("\n📝 STEP 1: GATHERING INPUTS")
        print("-" * 40)
        guided_inputs = self.guided_system.guided_input_workflow()
        
        # Step 2: Create forecast-ready DataFrame
        print("\n🔧 STEP 2: PREPARING DATA")
        print("-" * 40)
        forecast_df = self.guided_system.create_forecast_input(guided_inputs)
        
        # Step 3: Make predictions
        print("\n🔮 STEP 3: GENERATING FORECAST")
        print("-" * 40)
        
        if self.trained_model is not None:
            try:
                forecast_results = predict_future_arr(self.trained_model, forecast_df)
                forecast_success = True
            except Exception as e:
                print(f"⚠️ Model prediction failed: {e}")
                forecast_results = self._generate_fallback_forecast(guided_inputs)
                forecast_success = False
        else:
            forecast_results = self._generate_fallback_forecast(guided_inputs)
            forecast_success = False
        
        # Step 4: Generate insights
        print("\n📊 STEP 4: GENERATING INSIGHTS")
        print("-" * 40)
        insights = self._generate_insights(guided_inputs, forecast_results)
        
        # Step 5: Compile results
        results = {
            'company_name': guided_inputs['id_company'],
            'input_metrics': guided_inputs,
            'forecast_results': forecast_results,
            'insights': insights,
            'model_used': 'Trained Model' if forecast_success else 'Fallback Calculation',
            'forecast_success': forecast_success
        }
        
        return results
    
    def _generate_fallback_forecast(self, inputs: Dict) -> pd.DataFrame:
        """
        Generate a fallback forecast when the trained model is unavailable.
        
        Args:
            inputs: Dictionary with input metrics
            
        Returns:
            DataFrame with forecast results
        """
        print("📊 Generating fallback forecast using industry benchmarks...")
        
        carr = inputs['cARR']
        growth_rate = inputs['ARR YoY Growth (in %)']
        
        # Simple growth projection with slight deceleration
        quarters = ['FY26 Q1', 'FY26 Q2', 'FY26 Q3', 'FY26 Q4']
        growth_rates = []
        absolute_arr = []
        
        for i, quarter in enumerate(quarters):
            # Apply slight deceleration each quarter
            quarter_growth = growth_rate * (0.95 ** i)
            growth_rates.append(quarter_growth)
            
            # Calculate absolute ARR
            if i == 0:
                quarter_arr = carr * (1 + quarter_growth/100)
            else:
                quarter_arr = absolute_arr[-1] * (1 + quarter_growth/100)
            absolute_arr.append(quarter_arr)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Future Quarter': quarters,
            'Predicted YoY Growth (%)': growth_rates,
            'Predicted Absolute cARR (€)': absolute_arr
        })
        
        return forecast_df
    
    def _generate_insights(self, inputs: Dict, forecast: pd.DataFrame) -> Dict:
        """
        Generate business insights from the forecast results.
        
        Args:
            inputs: Dictionary with input metrics
            forecast: DataFrame with forecast results
            
        Returns:
            Dictionary with insights
        """
        insights = {}
        
        # Company size insights
        carr = inputs['cARR']
        if carr < 1e6:
            size_category = "Early Stage"
            size_insight = "You're in the early stage. Focus on product-market fit and customer acquisition."
        elif carr < 10e6:
            size_category = "Growth Stage"
            size_insight = "You're in the growth stage. Scale your sales and marketing efforts."
        elif carr < 100e6:
            size_category = "Scale Stage"
            size_insight = "You're in the scale stage. Optimize operations and expand internationally."
        else:
            size_category = "Enterprise"
            size_insight = "You're at enterprise scale. Focus on efficiency and market expansion."
        
        insights['size_category'] = size_category
        insights['size_insight'] = size_insight
        
        # Growth insights
        current_growth = inputs['ARR YoY Growth (in %)']
        if current_growth < 0:
            growth_insight = "⚠️ Your growth is negative. Focus on customer retention and product improvements."
        elif current_growth < 20:
            growth_insight = "📈 Your growth is modest. Consider increasing sales and marketing investment."
        elif current_growth < 50:
            growth_insight = "🚀 Your growth is strong. Focus on scaling operations efficiently."
        else:
            growth_insight = "🔥 Your growth is exceptional. Ensure you can sustain this pace."
        
        insights['growth_insight'] = growth_insight
        
        # Efficiency insights
        magic_number = inputs.get('Magic_Number', 0)
        if magic_number < 0.5:
            efficiency_insight = "⚠️ Your sales efficiency is low. Review your sales and marketing strategy."
        elif magic_number < 1.0:
            efficiency_insight = "📊 Your sales efficiency is moderate. There's room for optimization."
        else:
            efficiency_insight = "✅ Your sales efficiency is excellent. Keep up the great work!"
        
        insights['efficiency_insight'] = efficiency_insight
        
        # Forecast insights
        if not forecast.empty:
            avg_growth = forecast['Predicted YoY Growth (%)'].mean()
            if avg_growth > current_growth:
                forecast_insight = "📈 Your forecast shows improving growth trends."
            elif avg_growth < current_growth:
                forecast_insight = "📉 Your forecast shows declining growth. Consider strategic adjustments."
            else:
                forecast_insight = "➡️ Your forecast shows stable growth patterns."
            
            insights['forecast_insight'] = forecast_insight
            insights['avg_forecast_growth'] = avg_growth
        
        return insights
    
    def display_results(self, results: Dict):
        """
        Display the complete forecast results in a user-friendly format.
        
        Args:
            results: Dictionary with forecast results
        """
        print("\n" + "="*60)
        print(f"🎯 FORECAST RESULTS FOR {results['company_name'].upper()}")
        print("="*60)
        
        # Display input summary
        print(f"\n📝 INPUT SUMMARY:")
        print(f"  Current ARR: ${results['input_metrics']['cARR']:,.0f}")
        print(f"  Net New ARR: ${results['input_metrics']['Net New ARR']:,.0f}")
        print(f"  Growth Rate: {results['input_metrics']['ARR YoY Growth (in %)']:.1f}%")
        print(f"  Headcount: {results['input_metrics']['Headcount (HC)']:,} employees")
        
        # Display forecast
        if not results['forecast_results'].empty:
            print(f"\n🔮 4-QUARTER FORECAST:")
            print("-" * 50)
            print(f"{'Quarter':<12} {'YoY Growth':<12} {'Absolute ARR':<15}")
            print("-" * 50)
            
            for _, row in results['forecast_results'].iterrows():
                quarter = row['Future Quarter']
                growth = row['Predicted YoY Growth (%)']
                arr = row['Predicted Absolute cARR (€)']
                print(f"{quarter:<12} {growth:>8.1f}%    ${arr:>12,.0f}")
        
        # Display insights
        print(f"\n💡 BUSINESS INSIGHTS:")
        print("-" * 50)
        insights = results['insights']
        print(f"  Company Stage: {insights['size_category']}")
        print(f"  {insights['size_insight']}")
        print(f"  Growth Analysis: {insights['growth_insight']}")
        print(f"  Efficiency: {insights['efficiency_insight']}")
        
        if 'forecast_insight' in insights:
            print(f"  Forecast Trend: {insights['forecast_insight']}")
        
        # Display model info
        print(f"\n🔧 TECHNICAL DETAILS:")
        print("-" * 50)
        print(f"  Model Used: {results['model_used']}")
        print(f"  Forecast Success: {'✅ Yes' if results['forecast_success'] else '⚠️ Fallback'}")
        
        print(f"\n" + "="*60)
        print("✅ FORECAST COMPLETE!")
        print("="*60)
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save forecast results to a file.
        
        Args:
            results: Dictionary with forecast results
            filename: Optional filename, will auto-generate if not provided
        """
        if filename is None:
            company_name = results['company_name'].replace(' ', '_')
            filename = f"forecast_{company_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert DataFrame to dict for JSON serialization
        results_copy = results.copy()
        if not results_copy['forecast_results'].empty:
            results_copy['forecast_results'] = results_copy['forecast_results'].to_dict('records')
        
        # Save to JSON
        import json
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced predictor
    predictor = EnhancedFinancialPredictor()
    
    # Run the complete guided forecast
    results = predictor.run_guided_forecast()
    
    # Display results
    predictor.display_results(results)
    
    # Save results
    predictor.save_results(results) 