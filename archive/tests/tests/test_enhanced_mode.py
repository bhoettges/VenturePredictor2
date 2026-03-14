#!/usr/bin/env python3
"""
Test the enhanced mode functionality with sector, country, and currency selection
"""

import requests
import json

def test_basic_mode():
    """Test basic mode (minimal inputs only)."""
    print("🧪 TEST 1: BASIC MODE (Minimal Inputs)")
    print("=" * 50)
    
    payload = {
        "company_name": "Test Company Basic",
        "current_arr": 2800000,
        "net_new_arr": 800000,
        "enhanced_mode": False  # Basic mode
    }
    
    print(f"📊 Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:8000/guided_forecast", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Basic mode successful!")
            print(f"📈 Model used: {result.get('model_used', 'Unknown')}")
            print(f"📊 Forecast success: {result.get('forecast_success', False)}")
        else:
            print(f"❌ Basic mode failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing basic mode: {e}")

def test_enhanced_mode():
    """Test enhanced mode with sector, country, and currency selection."""
    print(f"\n🧪 TEST 2: ENHANCED MODE (With Sector/Country/Currency)")
    print("=" * 50)
    
    payload = {
        "company_name": "Test Company Enhanced",
        "current_arr": 2800000,
        "net_new_arr": 800000,
        "enhanced_mode": True,  # Enable enhanced mode
        "sector": "Cyber Security",  # Most common sector
        "country": "United States",  # Most common country
        "currency": "USD"  # Most common currency
    }
    
    print(f"📊 Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:8000/guided_forecast", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Enhanced mode successful!")
            print(f"📈 Model used: {result.get('model_used', 'Unknown')}")
            print(f"📊 Forecast success: {result.get('forecast_success', False)}")
        else:
            print(f"❌ Enhanced mode failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing enhanced mode: {e}")

def test_enhanced_mode_with_historical():
    """Test enhanced mode with historical ARR data."""
    print(f"\n🧪 TEST 3: ENHANCED MODE + HISTORICAL ARR")
    print("=" * 50)
    
    payload = {
        "company_name": "Test Company Enhanced Historical",
        "current_arr": 2800000,
        "net_new_arr": 800000,
        "enhanced_mode": True,
        "sector": "Data & Analytics",  # Second most common sector
        "country": "Israel",  # Second most common country
        "currency": "USD",
        "historical_arr": {
            "q1_arr": 1000000,
            "q2_arr": 1400000,
            "q3_arr": 2000000,
            "q4_arr": 2800000
        }
    }
    
    print(f"📊 Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:8000/guided_forecast", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Enhanced mode + historical successful!")
            print(f"📈 Model used: {result.get('model_used', 'Unknown')}")
            print(f"📊 Forecast success: {result.get('forecast_success', False)}")
        else:
            print(f"❌ Enhanced mode + historical failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing enhanced mode + historical: {e}")

def test_enhanced_mode_with_advanced():
    """Test enhanced mode with advanced metrics."""
    print(f"\n🧪 TEST 4: ENHANCED MODE + ADVANCED METRICS")
    print("=" * 50)
    
    payload = {
        "company_name": "Test Company Enhanced Advanced",
        "current_arr": 2800000,
        "net_new_arr": 800000,
        "enhanced_mode": True,
        "sector": "Infrastructure & Network",
        "country": "Germany",
        "currency": "EUR",
        "advanced_mode": True,
        "advanced_metrics": {
            "magic_number": 0.95,
            "gross_margin": 82.0,
            "headcount": 70,
            "sales_marketing": 1000000
        }
    }
    
    print(f"📊 Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post("http://localhost:8000/guided_forecast", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Enhanced mode + advanced successful!")
            print(f"📈 Model used: {result.get('model_used', 'Unknown')}")
            print(f"📊 Forecast success: {result.get('forecast_success', False)}")
        else:
            print(f"❌ Enhanced mode + advanced failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Error testing enhanced mode + advanced: {e}")

def test_validation_errors():
    """Test validation errors for invalid inputs."""
    print(f"\n🧪 TEST 5: VALIDATION ERRORS")
    print("=" * 50)
    
    # Test invalid sector
    payload_invalid_sector = {
        "company_name": "Test Company Invalid",
        "current_arr": 2800000,
        "net_new_arr": 800000,
        "enhanced_mode": True,
        "sector": "Invalid Sector",  # Invalid sector
        "country": "United States",
        "currency": "USD"
    }
    
    print(f"📊 Testing invalid sector:")
    print(json.dumps(payload_invalid_sector, indent=2))
    
    try:
        response = requests.post("http://localhost:8000/guided_forecast", json=payload_invalid_sector)
        if response.status_code == 422:  # Validation error
            print(f"✅ Validation error caught for invalid sector!")
            print(f"📝 Error: {response.json()}")
        else:
            print(f"❌ Expected validation error, got: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing validation: {e}")

def get_api_info():
    """Get API information and available options."""
    print(f"\n📋 API INFORMATION")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            info = response.json()
            print(f"📊 API Features:")
            for feature, description in info.get('features', {}).items():
                print(f"  • {feature}: {description}")
            
            print(f"\n📊 Enhanced Mode Options:")
            enhanced_options = info.get('enhanced_mode_options', {})
            print(f"  • Sectors: {enhanced_options.get('sectors', [])}")
            print(f"  • Countries: {enhanced_options.get('countries', [])}")
            print(f"  • Currencies: {enhanced_options.get('currencies', [])}")
            
            print(f"\n📊 Advanced Metrics:")
            print(f"  • Available metrics: {info.get('advanced_metrics', [])}")
        else:
            print(f"❌ Could not get API info: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting API info: {e}")

def main():
    """Run all enhanced mode tests."""
    print("🚀 TESTING ENHANCED MODE FUNCTIONALITY")
    print("=" * 70)
    
    # Get API information
    get_api_info()
    
    # Run tests
    test_basic_mode()
    test_enhanced_mode()
    test_enhanced_mode_with_historical()
    test_enhanced_mode_with_advanced()
    test_validation_errors()
    
    print(f"\n✅ Enhanced mode testing completed!")
    print(f"\n📋 Key Features Tested:")
    print(f"  • Basic mode (minimal inputs)")
    print(f"  • Enhanced mode (sector/country/currency)")
    print(f"  • Enhanced mode + historical ARR")
    print(f"  • Enhanced mode + advanced metrics")
    print(f"  • Input validation (error handling)")

if __name__ == "__main__":
    main()
