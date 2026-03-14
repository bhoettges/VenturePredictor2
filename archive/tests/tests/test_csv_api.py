#!/usr/bin/env python3
"""
Test the CSV Upload API Endpoint
===============================

Tests the new /predict_csv endpoint with a sample CSV file.
"""

import requests
import pandas as pd
import io

def create_test_csv():
    """Create a test CSV file with the required structure."""
    data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
        'ARR_End_of_Quarter': [1000000, 1400000, 2000000, 2800000],
        'Headcount': [95, 98, 100, 102],
        'Gross_Margin_Percent': [75, 76, 75, 75],
        'Net_Profit_Loss_Margin_Percent': [-15, -12, -10, -8],
        'Sales_Marketing': [400000, 500000, 600000, 800000],
        'Cash_Burn': [-300000, -350000, -400000, -500000],
        'Customers': [180, 190, 195, 200],
        'Churn_Rate': [0.05, 0.04, 0.05, 0.05],
        'Expansion_Rate': [0.10, 0.12, 0.15, 0.15]
    }
    
    df = pd.DataFrame(data)
    return df

def test_csv_api():
    """Test the CSV upload API endpoint."""
    base_url = "http://127.0.0.1:8000"
    
    print("TESTING CSV UPLOAD API ENDPOINT")
    print("=" * 50)
    
    # Create test CSV
    test_df = create_test_csv()
    print("Test CSV Data:")
    print(test_df.to_string(index=False))
    print()
    
    # Convert DataFrame to CSV string
    csv_buffer = io.StringIO()
    test_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Test 1: CSV Upload with company name
    print("1. Testing CSV Upload with Company Name")
    print("-" * 40)
    
    try:
        files = {
            'file': ('test_company.csv', csv_content, 'text/csv')
        }
        data = {
            'company_name': 'Test CSV Company'
        }
        
        response = requests.post(
            f"{base_url}/predict_csv",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Company: {result['company_name']}")
            print(f"Model: {result['model_used']}")
            print(f"Tier Used: {result['insights']['tier_used']}")
            print(f"Model Accuracy: {result['insights']['model_accuracy']}")
            
            print("\nCSV Info:")
            csv_info = result.get('csv_info', {})
            print(f"  Rows Processed: {csv_info.get('rows_processed', 'N/A')}")
            print(f"  Quarters Used: {csv_info.get('quarters_used', 'N/A')}")
            print(f"  Tier 2 Metrics Extracted: {csv_info.get('tier2_metrics_extracted', 'N/A')}")
            print(f"  CSV Columns: {csv_info.get('csv_columns', [])}")
            
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
    
    # Test 2: CSV Upload without company name
    print("\n\n2. Testing CSV Upload without Company Name")
    print("-" * 45)
    
    try:
        files = {
            'file': ('test_company.csv', csv_content, 'text/csv')
        }
        
        response = requests.post(
            f"{base_url}/predict_csv",
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Company: {result['company_name']}")
            print(f"Tier Used: {result['insights']['tier_used']}")
            
            print("\nForecast Results:")
            for forecast in result['forecast']:
                print(f"  {forecast['quarter']}: ${forecast['predicted_arr']:,.0f} "
                      f"(${forecast['pessimistic_arr']:,.0f} - ${forecast['optimistic_arr']:,.0f}) "
                      f"({forecast['yoy_growth_percent']:.1f}% YoY)")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    
    # Test 3: API Info
    print("\n\n3. Testing Updated API Info")
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
    test_csv_api()

