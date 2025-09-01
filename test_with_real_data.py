#!/usr/bin/env python3
"""
Test the API locally with real test data from test_company_2024.csv
"""

import requests
import json
import pandas as pd

# Local API URL
BASE_URL = "http://localhost:8000"

def test_with_real_csv_data():
    """Test the /predict_csv endpoint with the real test data"""
    print("🧪 Testing /predict_csv with real test data...")
    
    try:
        # Read the CSV file
        df = pd.read_csv('test_company_2024.csv')
        print(f"📊 Loaded CSV with {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        print(f"📊 Latest data: ARR=${df['ARR_End_of_Quarter'].iloc[-1]:,.0f}, Net New ARR=${df['Quarterly_Net_New_ARR'].iloc[-1]:,.0f}")
        
        # Test the CSV endpoint
        with open('test_company_2024.csv', 'rb') as f:
            files = {'file': ('test_company_2024.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/predict_csv", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CSV PREDICTION SUCCESS!")
            print(f"Company: {result.get('company_name')}")
            print(f"Model Used: {result.get('model_used')}")
            print(f"Forecast Success: {result.get('forecast_success')}")
            
            if 'forecast_results' in result:
                print("\n📊 Forecast Results:")
                for i, forecast in enumerate(result['forecast_results']):
                    print(f"  {forecast.get('Future Quarter', 'Q' + str(i+1))}: {forecast.get('Predicted YoY Growth (%)', 'N/A'):.1f}% growth")
                    if 'ARR_Realistic' in forecast:
                        print(f"    Expected ARR: ${forecast.get('ARR_Realistic', 0):,.0f}")
            
            return True
        else:
            print("❌ CSV PREDICTION FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_guided_forecast_with_latest_data():
    """Test the /guided_forecast endpoint with the latest data from CSV"""
    print("\n🧪 Testing /guided_forecast with latest data...")
    
    try:
        # Read the CSV and get latest data
        df = pd.read_csv('test_company_2024.csv')
        latest_row = df.iloc[-1]
        
        current_arr = int(latest_row['ARR_End_of_Quarter'])
        net_new_arr = int(latest_row['Quarterly_Net_New_ARR'])
        
        print(f"📊 Using latest data: ARR=${current_arr:,.0f}, Net New ARR=${net_new_arr:,.0f}")
        
        # Test data for guided forecast
        test_data = {
            "current_arr": current_arr,
            "net_new_arr": net_new_arr,
            "enhanced_mode": True,
            "sector": "Data & Analytics",
            "country": "United States",
            "currency": "USD"
        }
        
        response = requests.post(f"{BASE_URL}/guided_forecast", json=test_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ GUIDED FORECAST SUCCESS!")
            print(f"Company: {result.get('company_name')}")
            print(f"Model Used: {result.get('model_used')}")
            print(f"Forecast Success: {result.get('forecast_success')}")
            
            if 'forecast_results' in result:
                print("\n📊 Forecast Results:")
                for i, forecast in enumerate(result['forecast_results']):
                    print(f"  {forecast.get('Future Quarter', 'Q' + str(i+1))}: {forecast.get('Predicted YoY Growth (%)', 'N/A'):.1f}% growth")
                    if 'ARR_Realistic' in forecast:
                        print(f"    Expected ARR: ${forecast.get('ARR_Realistic', 0):,.0f}")
            
            return True
        else:
            print("❌ GUIDED FORECAST FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def test_chat_with_real_data():
    """Test the chat endpoint with real data"""
    print("\n🧪 Testing /chat with real data...")
    
    try:
        # Read the CSV and get latest data
        df = pd.read_csv('test_company_2024.csv')
        latest_row = df.iloc[-1]
        
        current_arr = int(latest_row['ARR_End_of_Quarter'])
        net_new_arr = int(latest_row['Quarterly_Net_New_ARR'])
        
        # Format as natural language
        current_arr_m = current_arr / 1000000
        net_new_arr_k = net_new_arr / 1000
        
        test_data = {
            "message": f"My ARR is ${current_arr_m:.1f}M and net new ARR is ${net_new_arr_k:.0f}K, can you forecast my growth?",
            "name": "Test User"
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=test_data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CHAT SUCCESS!")
            print(f"Response: {result.get('response', '')[:200]}...")
            return True
        else:
            print("❌ CHAT FAILED!")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing API with Real Test Data")
    print("=" * 60)
    
    # Test CSV endpoint
    csv_success = test_with_real_csv_data()
    
    # Test guided forecast
    guided_success = test_guided_forecast_with_latest_data()
    
    # Test chat (might fail due to xlrd issue)
    chat_success = test_chat_with_real_data()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print(f"  CSV Prediction: {'✅ PASS' if csv_success else '❌ FAIL'}")
    print(f"  Guided Forecast: {'✅ PASS' if guided_success else '❌ FAIL'}")
    print(f"  Chat: {'✅ PASS' if chat_success else '❌ FAIL'}")
    
    if csv_success and guided_success:
        print("\n🎉 CORE FUNCTIONALITY WORKING! Ready for deployment!")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")
