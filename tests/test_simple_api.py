#!/usr/bin/env python3
"""
Simple test script for the guided_forecast endpoint only
"""

import requests
import json

# Local API URL
BASE_URL = "http://localhost:8000"

def test_guided_forecast_simple():
    """Test the guided_forecast endpoint with minimal data"""
    print("🧪 Testing /guided_forecast endpoint (simple)...")
    
    # Minimal test data
    test_data = {
        "current_arr": 2100000,
        "net_new_arr": 320000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/guided_forecast", json=test_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Company: {result.get('company_name')}")
            print(f"Model Used: {result.get('model_used')}")
            print(f"Forecast Success: {result.get('forecast_success')}")
            
            if 'forecast_results' in result:
                print("\n📊 Forecast Results:")
                for i, forecast in enumerate(result['forecast_results']):
                    print(f"  Q{i+1}: {forecast.get('Predicted YoY Growth (%)', 'N/A'):.1f}% growth")
            
            return True
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_guided_forecast_enhanced():
    """Test the guided_forecast endpoint with enhanced mode"""
    print("\n🧪 Testing /guided_forecast endpoint (enhanced)...")
    
    # Enhanced test data
    test_data = {
        "current_arr": 7800000,
        "net_new_arr": 1100000,
        "enhanced_mode": True,
        "sector": "Data & Analytics",
        "country": "United States",
        "currency": "USD"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/guided_forecast", json=test_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Company: {result.get('company_name')}")
            print(f"Model Used: {result.get('model_used')}")
            print(f"Forecast Success: {result.get('forecast_success')}")
            
            if 'forecast_results' in result:
                print("\n📊 Forecast Results:")
                for i, forecast in enumerate(result['forecast_results']):
                    print(f"  Q{i+1}: {forecast.get('Predicted YoY Growth (%)', 'N/A'):.1f}% growth")
            
            return True
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Local API - Guided Forecast Only")
    print("=" * 50)
    
    # Test simple forecast
    test_guided_forecast_simple()
    
    # Test enhanced forecast
    test_guided_forecast_enhanced()
    
    print("\n" + "=" * 50)
    print("✅ Simple testing complete!")
