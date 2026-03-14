#!/usr/bin/env python3
"""
Test script to verify the /guided_forecast API endpoint works.
"""

import requests
import json

def test_guided_forecast():
    """Test the guided_forecast endpoint with real data."""
    
    print("🧪 Testing /guided_forecast API endpoint...")
    
    # Test data (your real 2024 Q4 data)
    test_data = {
        "company_name": "Test Company 2024",
        "current_arr": 2100000,      # Q4 2024 ARR
        "net_new_arr": 320000,       # Q4 2024 Net New ARR
        "growth_rate": 18.0,         # Calculated growth rate
        "advanced_mode": False,
        "advanced_metrics": None
    }
    
    try:
        # Make API call
        print(f"📤 Sending request to /guided_forecast...")
        print(f"📊 Test data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(
            "http://localhost:8000/guided_forecast",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! API endpoint working.")
            
            print(f"\n📈 FORECAST RESULTS:")
            print(f"Company: {result['company_name']}")
            print(f"Model Used: {result['model_used']}")
            print(f"Forecast Success: {result['forecast_success']}")
            
            if result['forecast_results']:
                print(f"\n🔮 4-Quarter Forecast:")
                for i, quarter in enumerate(result['forecast_results']):
                    print(f"  Q{i+1}: {quarter['Predicted YoY Growth (%)']:.1f}% → ${quarter['Predicted Absolute cARR (€)']:,.0f}")
            
            if result['insights']:
                print(f"\n💡 INSIGHTS:")
                print(f"  Stage: {result['insights']['size_category']}")
                print(f"  Growth: {result['insights']['growth_insight']}")
                print(f"  Efficiency: {result['insights']['efficiency_insight']}")
            
            return True
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Could not connect to server")
        print("Make sure the FastAPI server is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_guided_forecast()
    if success:
        print("\n🎉 API endpoint test completed successfully!")
        print("Your frontend can now use this endpoint!")
    else:
        print("\n💥 API endpoint test failed!")
        print("Check the server logs for more details.")
