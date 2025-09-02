#!/usr/bin/env python3
"""
Demo script for the Guided Input with Intelligent Defaults system.
This showcases the enhanced financial forecasting system we discussed.
"""

import sys
import os
from enhanced_prediction import EnhancedFinancialPredictor

def main():
    """
    Main demo function that showcases the guided input system.
    """
    print("🎯 GUIDED FINANCIAL FORECASTING DEMO")
    print("=" * 60)
    print("This demo showcases the enhanced forecasting system with intelligent defaults.")
    print("You'll be asked for just a few key metrics, and the system will intelligently")
    print("estimate the rest based on patterns from similar companies.\n")
    
    # Initialize the enhanced predictor
    try:
        predictor = EnhancedFinancialPredictor()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Run the guided forecast
    try:
        print("\n🚀 Starting guided forecast workflow...")
        results = predictor.run_guided_forecast()
        
        # Display results
        predictor.display_results(results)
        
        # Ask if user wants to save results
        save_choice = input("\n💾 Would you like to save the results? (y/N): ").strip().lower()
        if save_choice == 'y':
            predictor.save_results(results)
        
        # Ask if user wants to run another forecast
        another_choice = input("\n🔄 Would you like to run another forecast? (y/N): ").strip().lower()
        if another_choice == 'y':
            print("\n" + "="*60)
            main()  # Recursive call for another forecast
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during forecast: {e}")
        print("The system will use fallback calculations.")

def quick_demo():
    """
    Quick demo with predefined inputs for testing.
    """
    print("🎯 QUICK DEMO - Using Sample Data")
    print("=" * 50)
    
    # Sample company data
    sample_inputs = {
        'id_company': 'Demo Company',
        'cARR': 5000000,  # $5M ARR
        'Net New ARR': 750000,  # $750K new ARR
        'ARR YoY Growth (in %)': 15.0,  # 15% growth
        'Quarter Num': 1
    }
    
    try:
        from guided_input_system import GuidedInputSystem
        
        # Initialize guided system
        guided_system = GuidedInputSystem()
        guided_system.initialize_from_training_data()
        
        # Infer secondary metrics
        inferred_metrics = guided_system.infer_secondary_metrics(sample_inputs)
        
        # Display results
        print(f"\n📊 SAMPLE COMPANY: {sample_inputs['id_company']}")
        print("-" * 50)
        print(f"Primary Inputs:")
        print(f"  Current ARR: ${sample_inputs['cARR']:,.0f}")
        print(f"  Net New ARR: ${sample_inputs['Net New ARR']:,.0f}")
        print(f"  Growth Rate: {sample_inputs['ARR YoY Growth (in %)']:.1f}%")
        
        print(f"\nIntelligent Defaults:")
        print(f"  Headcount: {inferred_metrics['Headcount (HC)']:,} employees")
        print(f"  Sales & Marketing: ${inferred_metrics['Sales & Marketing']:,.0f}")
        print(f"  Cash Burn: ${inferred_metrics['Cash Burn (OCF & ICF)']:,.0f}")
        print(f"  Gross Margin: {inferred_metrics['Gross Margin (in %)']:.1f}%")
        print(f"  Customers: {inferred_metrics['Customers (EoP)']:,}")
        
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in quick demo: {e}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        main() 