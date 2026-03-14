#!/usr/bin/env python3
"""Test enhanced mode functionality in the chat endpoint."""

import requests
import json

def test_enhanced_chat():
    """Test chat endpoint with enhanced mode parameters."""
    
    # Test basic chat
    print("🧪 Testing basic chat functionality...")
    basic_response = requests.post(
        "http://localhost:8000/chat",
        json={
            "message": "My ARR is $2.8M and net new ARR is $800K. Can you forecast my growth?",
            "name": "Test User"
        }
    )
    
    if basic_response.status_code == 200:
        print("✅ Basic chat works!")
        print(f"Response: {basic_response.json()['response'][:100]}...")
    else:
        print(f"❌ Basic chat failed: {basic_response.status_code}")
    
    # Test enhanced mode chat
    print("\n🧪 Testing enhanced mode chat functionality...")
    enhanced_response = requests.post(
        "http://localhost:8000/chat",
        json={
            "message": "My ARR is $2.8M and net new ARR is $800K, sector: Data & Analytics, country: United States, currency: USD. Can you forecast my growth?",
            "name": "Test User"
        }
    )
    
    if enhanced_response.status_code == 200:
        print("✅ Enhanced chat works!")
        response_data = enhanced_response.json()
        print(f"Response: {response_data['response'][:100]}...")
        
        # Check if enhanced data is included
        if 'data' in response_data:
            print("✅ Enhanced data structure present!")
        else:
            print("⚠️ Enhanced data structure missing")
    else:
        print(f"❌ Enhanced chat failed: {enhanced_response.status_code}")
        print(f"Error: {enhanced_response.text}")

if __name__ == "__main__":
    test_enhanced_chat()
