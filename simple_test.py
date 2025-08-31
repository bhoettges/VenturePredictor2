#!/usr/bin/env python3
"""
Simple test to isolate the Sales & Marketing column issue
"""

import requests
import json

# Simple test payload
payload = {
    "company_name": "Test Company",
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
    "advanced_metrics": {}  # Empty dict instead of None
}

try:
    response = requests.post("http://localhost:8000/predict", json=payload, timeout=30)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS!")
        print(f"Company: {result.get('company_name')}")
        print(f"Model Used: {result.get('model_used')}")
        print(f"Message: {result.get('message')}")
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {str(e)}")
