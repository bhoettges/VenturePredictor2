#!/usr/bin/env python3
"""
Test script for the new unified /predict endpoint
Tests both basic mode and advanced mode functionality
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:8000"  # Change this to your deployed URL if testing production
ENDPOINT = "/predict"

def test_basic_mode():
    """Test basic forecasting mode (advanced_mode: false)"""
    print("🧪 Testing Basic Mode (Advanced Mode OFF)")
    print("=" * 50)
    
    payload = {
        "company_name": "TechStartup Inc",
        "q1_arr": 1000000,
        "q1_net_new_arr": 200000,
        "q1_qrr": 250000,
        "q1_headcount": 50,
        "q1_gross_margin": 75.0,
        "q1_net_profit_loss": -5.0,
        "q2_arr": 1200000,
        "q2_net_new_arr": 250000,
        "q2_qrr": 300000,
        "q2_headcount": 60,
        "q2_gross_margin": 78.0,
        "q2_net_profit_loss": -3.0,
        "q3_arr": 1450000,
        "q3_net_new_arr": 300000,
        "q3_qrr": 362500,
        "q3_headcount": 70,
        "q3_gross_margin": 80.0,
        "q3_net_profit_loss": -1.0,
        "q4_arr": 1750000,
        "q4_net_new_arr": 350000,
        "q4_qrr": 437500,
        "q4_headcount": 80,
        "q4_gross_margin": 82.0,
        "q4_net_profit_loss": 2.0,
        "advanced_mode": False,
        "advanced_metrics": None
    }
    
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, timeout=30)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Company: {result.get('company_name')}")
            print(f"✅ Model Used: {result.get('model_used')}")
            print(f"✅ Forecast Success: {result.get('forecast_success')}")
            print(f"✅ Advanced Mode: {result.get('advanced_mode_enabled')}")
            print(f"✅ Message: {result.get('message')}")
            
            # Check insights
            insights = result.get('insights', {})
            print(f"✅ Size Category: {insights.get('size_category')}")
            print(f"✅ Growth Insight: {insights.get('growth_insight')}")
            print(f"✅ Efficiency Insight: {insights.get('efficiency_insight')}")
            
            # Check forecast results
            forecast_results = result.get('forecast_results', [])
            if forecast_results:
                print(f"✅ Forecast Results: {len(forecast_results)} periods")
                print(f"✅ Sample Forecast: {forecast_results[0] if forecast_results else 'None'}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_advanced_mode():
    """Test advanced forecasting mode (advanced_mode: true) with metric overrides"""
    print("\n🧪 Testing Advanced Mode (Advanced Mode ON)")
    print("=" * 50)
    
    payload = {
        "company_name": "AdvancedTech Corp",
        "q1_arr": 2000000,
        "q1_net_new_arr": 400000,
        "q1_qrr": 500000,
        "q1_headcount": 100,
        "q1_gross_margin": 80.0,
        "q1_net_profit_loss": -2.0,
        "q2_arr": 2400000,
        "q2_net_new_arr": 500000,
        "q2_qrr": 600000,
        "q2_headcount": 120,
        "q2_gross_margin": 82.0,
        "q2_net_profit_loss": 1.0,
        "q3_arr": 2900000,
        "q3_net_new_arr": 600000,
        "q3_qrr": 725000,
        "q3_headcount": 140,
        "q3_gross_margin": 85.0,
        "q3_net_profit_loss": 3.0,
        "q4_arr": 3500000,
        "q4_net_new_arr": 700000,
        "q4_qrr": 875000,
        "q4_headcount": 160,
        "q4_gross_margin": 87.0,
        "q4_net_profit_loss": 5.0,
        "advanced_mode": True,
        "advanced_metrics": {
            "q1": {
                "sales_marketing": 800000,     # User override
                "cash_burn": -200000,          # User override (negative for cash burn)
                "ebitda": -300000              # User override
            },
            "q2": {
                "gross_margin": 85.0,          # User override
                "sales_marketing": 0,          # 0 = estimate automatically
                "ebitda": -400000              # User override
            },
            "q3": {
                "sales_marketing": 900000,     # User override
                "cash_burn": -250000           # User override
            },
            "q4": {
                "sales_marketing": 0,          # 0 = estimate automatically
                "ebitda": -100000              # User override
            },
            "global": {
                "magic_number_override": 0.8,  # Global override for all quarters
                "burn_multiple_override": 0     # 0 = estimate automatically
            }
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, timeout=30)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Company: {result.get('company_name')}")
            print(f"✅ Model Used: {result.get('model_used')}")
            print(f"✅ Forecast Success: {result.get('forecast_success')}")
            print(f"✅ Advanced Mode: {result.get('advanced_mode_enabled')}")
            print(f"✅ Advanced Metrics Count: {result.get('advanced_metrics_count')}")
            print(f"✅ Message: {result.get('message')}")
            
            # Check insights
            insights = result.get('insights', {})
            print(f"✅ Size Category: {insights.get('size_category')}")
            print(f"✅ Growth Insight: {insights.get('growth_insight')}")
            print(f"✅ Efficiency Insight: {insights.get('efficiency_insight')}")
            
            # Check advanced mode specific insights
            if insights.get('advanced_mode_info'):
                print(f"✅ Advanced Mode Info: {insights.get('advanced_mode_info')}")
            
            # Check forecast results
            forecast_results = result.get('forecast_results', [])
            if forecast_results:
                print(f"✅ Forecast Results: {len(forecast_results)} periods")
                print(f"✅ Sample Forecast: {forecast_results[0] if forecast_results else 'None'}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_error_handling():
    """Test error handling with invalid data"""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    # Test with missing required fields
    payload = {
        "company_name": "Invalid Company",
        "q1_arr": 1000000,
        # Missing other required fields
        "advanced_mode": False
    }
    
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, timeout=30)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 422:  # Validation error
            print("✅ Correctly caught validation error for missing fields")
            return True
        else:
            print(f"❌ Unexpected response: {response.status_code}")
            print(f"❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Unified /predict Endpoint")
    print("=" * 60)
    
    # Check if server is running
    try:
        health_check = requests.get(f"{BASE_URL}/", timeout=5)
        if health_check.status_code == 200:
            print("✅ Server is running and accessible")
        else:
            print(f"⚠️ Server responded with status: {health_check.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        print("💡 Make sure the FastAPI server is running on the specified URL")
        return
    
    # Run tests
    tests = [
        ("Basic Mode", test_basic_mode),
        ("Advanced Mode", test_advanced_mode),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The /predict endpoint is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
