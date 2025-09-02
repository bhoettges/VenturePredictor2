#!/usr/bin/env python3
"""
Test the new Tier-Based API Endpoint
====================================

Tests the new /tier_based_forecast endpoint with both Tier 1 only and Tier 1+2 scenarios.
"""

import requests
import json

def test_tier_based_api():
    """Test the tier-based API endpoint."""
    base_url = "http://127.0.0.1:8000"
    
    # Test data - Tier 1 only
    tier1_only_request = {
        "company_name": "Test Company",
        "q1_arr": 1000000,    # $1M
        "q2_arr": 1400000,    # $1.4M
        "q3_arr": 2000000,    # $2M
        "q4_arr": 2800000,    # $2.8M
        "headcount": 102,
        "sector": "Data & Analytics"
    }
    
    # Test data - Tier 1 + Tier 2
    tier1_plus_tier2_request = {
        "company_name": "Test Company Advanced",
        "q1_arr": 1000000,    # $1M
        "q2_arr": 1400000,    # $1.4M
        "q3_arr": 2000000,    # $2M
        "q4_arr": 2800000,    # $2.8M
        "headcount": 102,
        "sector": "Data & Analytics",
        "tier2_metrics": {
            "gross_margin": 75,
            "sales_marketing": 1200000,  # $1.2M
            "cash_burn": -800000,        # -$800K
            "customers": 200,
            "churn_rate": 0.05,          # 5%
            "expansion_rate": 0.15       # 15%
        }
    }
    
    print("TESTING TIER-BASED API ENDPOINT")
    print("=" * 50)
    
    # Test 1: Tier 1 Only
    print("\n1. Testing Tier 1 Only (Required Input)")
    print("-" * 40)
    
    try:
        response = requests.post(
            f"{base_url}/tier_based_forecast",
            json=tier1_only_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Company: {result['company_name']}")
            print(f"Model: {result['model_used']}")
            print(f"Tier Used: {result['insights']['tier_used']}")
            print(f"Model Accuracy: {result['insights']['model_accuracy']}")
            
            print("\nForecast Results:")
            for forecast in result['forecast']:
                print(f"  {forecast['quarter']}: ${forecast['predicted_arr']:,.0f} "
                      f"(${forecast['pessimistic_arr']:,.0f} - ${forecast['optimistic_arr']:,.0f}) "
                      f"({forecast['yoy_growth_percent']:.1f}% YoY)")
            
            print(f"\nTotal Growth: {result['insights']['total_growth_percent']:.1f}%")
            print(f"Tier Analysis: {result['tier_analysis']}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Make sure the API server is running on http://127.0.0.1:8000")
        return
    
    # Test 2: Tier 1 + Tier 2
    print("\n\n2. Testing Tier 1 + Tier 2 (Advanced Analysis)")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{base_url}/tier_based_forecast",
            json=tier1_plus_tier2_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Company: {result['company_name']}")
            print(f"Model: {result['model_used']}")
            print(f"Tier Used: {result['insights']['tier_used']}")
            print(f"Model Accuracy: {result['insights']['model_accuracy']}")
            
            print("\nForecast Results:")
            for forecast in result['forecast']:
                print(f"  {forecast['quarter']}: ${forecast['predicted_arr']:,.0f} "
                      f"(${forecast['pessimistic_arr']:,.0f} - ${forecast['optimistic_arr']:,.0f}) "
                      f"({forecast['yoy_growth_percent']:.1f}% YoY)")
            
            print(f"\nTotal Growth: {result['insights']['total_growth_percent']:.1f}%")
            print(f"Tier Analysis: {result['tier_analysis']}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    
    # Test 3: API Info
    print("\n\n3. Testing API Info Endpoint")
    print("-" * 30)
    
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ API Info Retrieved!")
            print(f"Version: {result['version']}")
            print(f"Status: {result['status']}")
            print(f"Model Accuracy: {result['model_accuracy']}")
            
            print("\nAvailable Endpoints:")
            for endpoint, description in result['endpoints'].items():
                print(f"  {endpoint}: {description}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_tier_based_api()

