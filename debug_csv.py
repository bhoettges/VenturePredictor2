#!/usr/bin/env python3
"""
Debug CSV processing
"""

import requests
import pandas as pd
import io

# Create test CSV
data = {
    'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'],
    'ARR_End_of_Quarter': [1000000, 1400000, 2000000, 2800000],
    'Headcount': [95, 98, 100, 102],
    'Gross_Margin_Percent': [75, 76, 75, 75]
}

df = pd.DataFrame(data)
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue()

print("Test CSV:")
print(csv_content)

# Test the endpoint
files = {
    'file': ('test.csv', csv_content, 'text/csv')
}

try:
    response = requests.post(
        "http://127.0.0.1:8000/predict_csv",
        files=files
    )
    
    print(f"\nResponse Status: {response.status_code}")
    print(f"Response Content: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")

