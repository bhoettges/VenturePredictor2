#!/usr/bin/env python3
"""
Test the new advanced mode capabilities in the FastAPI guided forecast endpoint.
"""

import requests
import json

def test_basic_guided_forecast():
    """Test basic guided forecast (minimal inputs)."""
    print("🎯 TESTING BASIC GUIDED FORECAST")
    print("=" * 50)
    
    url = "http://localhost:8000/guided_forecast"
    
    # Basic request - only minimal inputs
    basic_request = {
        "company_name": "Test Company 2024",
        "current_arr": 2100000,
        "net_new_arr": 320000,
        "advanced_mode": False
    }
    
    try:
        response = requests.post(url, json=basic_request)
        if response.status_code == 200:
            result = response.json()
            print("✅ Basic forecast successful!")
            print(f"Model used: {result['model_used']}")
            print(f"Forecast success: {result['forecast_success']}")
            
            # Show some key inferred metrics
            metrics = result['input_metrics']
            print(f"\n📊 Key Inferred Metrics:")
            print(f"  Sales & Marketing: ${metrics.get('Sales & Marketing', 0):,.0f}")
            print(f"  EBITDA: ${metrics.get('EBITDA', 0):,.0f}")
            print(f"  Magic Number: {metrics.get('Magic_Number', 0):.2f}")
            print(f"  Gross Margin: {metrics.get('Gross Margin (in %)', 0):.1f}%")
            
            # Show forecast results
            if 'forecast_results' in result and result['forecast_results']:
                print(f"\n🔮 Forecast Results:")
                for i, forecast in enumerate(result['forecast_results'][:2]):  # Show first 2 quarters
                    if 'Future Quarter' in forecast:
                        print(f"  {forecast['Future Quarter']}: {forecast.get('Realistic', forecast.get('Predicted YoY Growth (%)', 0)):.1f}% growth")
            
            return True
        else:
            print(f"❌ Basic forecast failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error testing basic forecast: {e}")
        return False

def test_advanced_mode_forecast():
    """Test advanced mode with custom overrides."""
    print("\n🎯 TESTING ADVANCED MODE FORECAST")
    print("=" * 50)
    
    url = "http://localhost:8000/guided_forecast"
    
    # Advanced request with custom overrides
    advanced_request = {
        "company_name": "Test Company 2024 Advanced",
        "current_arr": 2100000,
        "net_new_arr": 320000,
        "advanced_mode": True,
        "advanced_metrics": {
            "sales_marketing": 1500000,      # Override Sales & Marketing
            "ebitda": 400000,               # Override EBITDA
            "cash_burn": -300000,           # Override Cash Burn
            "magic_number": 0.85,           # Override Magic Number
            "gross_margin": 78.5,           # Override Gross Margin
            "headcount": 45,                # Override Headcount
            "customers_eop": 250,           # Override Customers
            "expansion_upsell": 180000,     # Override Expansion & Upsell
            "churn_reduction": -50000       # Override Churn & Reduction
        }
    }
    
    try:
        response = requests.post(url, json=advanced_request)
        if response.status_code == 200:
            result = response.json()
            print("✅ Advanced forecast successful!")
            print(f"Model used: {result['model_used']}")
            print(f"Forecast success: {result['forecast_success']}")
            
            # Show overridden metrics
            metrics = result['input_metrics']
            print(f"\n🔧 Overridden Metrics:")
            print(f"  Sales & Marketing: ${metrics.get('Sales & Marketing', 0):,.0f} (overridden)")
            print(f"  EBITDA: ${metrics.get('EBITDA', 0):,.0f} (overridden)")
            print(f"  Magic Number: {metrics.get('Magic_Number', 0):.2f} (overridden)")
            print(f"  Gross Margin: {metrics.get('Gross Margin (in %)', 0):.1f}% (overridden)")
            print(f"  Headcount: {metrics.get('Headcount (HC)', 0):.0f} (overridden)")
            print(f"  Customers: {metrics.get('Customers (EoP)', 0):.0f} (overridden)")
            
            # Show forecast results with uncertainty
            if 'forecast_results' in result and result['forecast_results']:
                print(f"\n🔮 Forecast Results (with ±10% uncertainty):")
                for i, forecast in enumerate(result['forecast_results'][:2]):  # Show first 2 quarters
                    if 'Future Quarter' in forecast:
                        quarter = forecast['Future Quarter']
                        realistic = forecast.get('Realistic', forecast.get('Predicted YoY Growth (%)', 0))
                        pessimistic = forecast.get('Pessimistic', realistic)
                        optimistic = forecast.get('Optimistic', realistic)
                        print(f"  {quarter}: {realistic:.1f}% ({pessimistic:.1f}% to {optimistic:.1f}%)")
            
            return True
        else:
            print(f"❌ Advanced forecast failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error testing advanced forecast: {e}")
        return False

def test_partial_advanced_mode():
    """Test advanced mode with only some metrics overridden."""
    print("\n🎯 TESTING PARTIAL ADVANCED MODE")
    print("=" * 50)
    
    url = "http://localhost:8000/guided_forecast"
    
    # Partial advanced request - only override a few key metrics
    partial_request = {
        "company_name": "Test Company 2024 Partial",
        "current_arr": 2100000,
        "net_new_arr": 320000,
        "advanced_mode": True,
        "advanced_metrics": {
            "magic_number": 0.95,           # Only override Magic Number
            "gross_margin": 82.0,           # Only override Gross Margin
            "headcount": 38                 # Only override Headcount
        }
    }
    
    try:
        response = requests.post(url, json=partial_request)
        if response.status_code == 200:
            result = response.json()
            print("✅ Partial advanced forecast successful!")
            print(f"Model used: {result['model_used']}")
            
            # Show what was overridden vs inferred
            metrics = result['input_metrics']
            print(f"\n🔧 Overridden Metrics:")
            print(f"  Magic Number: {metrics.get('Magic_Number', 0):.2f} (overridden)")
            print(f"  Gross Margin: {metrics.get('Gross Margin (in %)', 0):.1f}% (overridden)")
            print(f"  Headcount: {metrics.get('Headcount (HC)', 0):.0f} (overridden)")
            
            print(f"\n📊 Inferred Metrics (smart defaults):")
            print(f"  Sales & Marketing: ${metrics.get('Sales & Marketing', 0):,.0f} (inferred)")
            print(f"  EBITDA: ${metrics.get('EBITDA', 0):,.0f} (inferred)")
            print(f"  Cash Burn: ${metrics.get('Cash Burn (OCF & ICF)', 0):,.0f} (inferred)")
            
            return True
        else:
            print(f"❌ Partial advanced forecast failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error testing partial advanced forecast: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING ADVANCED MODE CAPABILITIES")
    print("=" * 60)
    
    # Test all three scenarios
    test_basic_guided_forecast()
    test_advanced_mode_forecast()
    test_partial_advanced_mode()
    
    print("\n✅ Advanced mode testing completed!")
    print("\n📋 Available Advanced Metrics:")
    print("  - sales_marketing: Sales & Marketing spend")
    print("  - ebitda: EBITDA")
    print("  - cash_burn: Cash Burn (OCF & ICF)")
    print("  - rule_of_40: LTM Rule of 40% (ARR)")
    print("  - arr_yoy_growth: ARR YoY Growth (%)")
    print("  - revenue_yoy_growth: Revenue YoY Growth (%)")
    print("  - magic_number: Magic Number")
    print("  - burn_multiple: Burn Multiple")
    print("  - customers_eop: Customers (End of Period)")
    print("  - expansion_upsell: Expansion & Upsell")
    print("  - churn_reduction: Churn & Reduction")
    print("  - gross_margin: Gross Margin (%)")
    print("  - headcount: Headcount")
    print("  - net_profit_margin: Net Profit/Loss Margin (%)")
