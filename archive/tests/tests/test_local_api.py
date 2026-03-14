#!/usr/bin/env python3
"""
Test script for local API endpoints
"""

import requests
import json

# Local API URL
BASE_URL = "http://localhost:8000"

def test_guided_forecast():
    """Test the guided_forecast endpoint"""
    print("🧪 Testing /guided_forecast endpoint...")
    
    # Test data
    test_data = {
        "current_arr": 2100000,
        "net_new_arr": 320000,
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
                for i, forecast in enumerate(result['forecast_results'][:2]):  # Show first 2 quarters
                    print(f"  Q{i+1}: {forecast.get('Predicted YoY Growth (%)', 'N/A'):.1f}% growth")
            
            return True
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🧪 Testing / endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Message: {result.get('message')}")
            print(f"Endpoints: {list(result.get('endpoints', {}).keys())}")
            return True
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_chat_endpoint():
    """Test the chat endpoint"""
    print("\n🧪 Testing /chat endpoint...")
    
    test_data = {
        "message": "My ARR is $2.1M and net new ARR is $320K, can you forecast my growth?",
        "name": "Test User"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=test_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Response: {result.get('response', '')[:100]}...")
            return True
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Local API Endpoints")
    print("=" * 50)
    
    # Test root endpoint first
    test_root_endpoint()
    
    # Test guided forecast
    test_guided_forecast()
    
    # Test chat endpoint
    test_chat_endpoint()
    
    print("\n" + "=" * 50)
    print("✅ Local testing complete!")
