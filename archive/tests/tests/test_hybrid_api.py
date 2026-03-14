#!/usr/bin/env python3
"""
Test the hybrid prediction API with different company scenarios
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.services.tier_prediction import perform_tier_based_forecast
from api.models.schemas import TierBasedRequest
import json

def test_declining_company():
    """Test API with declining company."""
    
    print("=" * 80)
    print("TEST 1: DECLINING COMPANY")
    print("=" * 80)
    
    request = TierBasedRequest(
        company_name="Struggling SaaS Inc",
        q1_arr=2000000,
        q2_arr=1500000,
        q3_arr=1000000,
        q4_arr=500000,
        headcount=50,
        sector="Data & Analytics"
    )
    
    result = perform_tier_based_forecast(request)
    
    if result['success']:
        print(f"\n✅ Success!")
        print(f"Company: {result['company_name']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Prediction Method: {result['prediction_method']}")
        
        print(f"\n📊 Trend Analysis:")
        trend = result['trend_analysis']
        print(f"  Type: {trend['trend_type']}")
        print(f"  Confidence: {trend['confidence']}")
        print(f"  Message: {trend['user_message']}")
        print(f"  Reason: {trend['reason']}")
        
        if 'gpt_analysis' in result:
            print(f"\n💡 GPT Analysis:")
            gpt = result['gpt_analysis']
            print(f"  Reasoning: {gpt['reasoning']}")
            print(f"  Confidence: {gpt['confidence']}")
            print(f"  Key Assumption: {gpt['key_assumption']}")
        
        print(f"\n📈 Predictions:")
        for forecast in result['forecast']:
            print(f"  {forecast['quarter']}: ${forecast['predicted_arr']:,.0f} "
                  f"(YoY: {forecast['yoy_growth_percent']:+.1f}%, "
                  f"QoQ: {forecast['qoq_growth_percent']:+.1f}%)")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

def test_growing_company():
    """Test API with growing company."""
    
    print("\n" + "=" * 80)
    print("TEST 2: GROWING COMPANY")
    print("=" * 80)
    
    request = TierBasedRequest(
        company_name="Rocket Growth Co",
        q1_arr=1000000,
        q2_arr=1400000,
        q3_arr=2000000,
        q4_arr=2800000,
        headcount=100,
        sector="Cyber Security"
    )
    
    result = perform_tier_based_forecast(request)
    
    if result['success']:
        print(f"\n✅ Success!")
        print(f"Company: {result['company_name']}")
        print(f"Model Used: {result['model_used']}")
        print(f"Prediction Method: {result['prediction_method']}")
        
        print(f"\n📊 Trend Analysis:")
        trend = result['trend_analysis']
        print(f"  Type: {trend['trend_type']}")
        print(f"  Confidence: {trend['confidence']}")
        print(f"  Message: {trend['user_message']}")
        
        print(f"\n📈 Predictions:")
        for forecast in result['forecast'][:2]:  # Show first 2
            print(f"  {forecast['quarter']}: ${forecast['predicted_arr']:,.0f} "
                  f"(YoY: {forecast['yoy_growth_percent']:+.1f}%)")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

def test_v_shape_recovery():
    """Test API with V-shape recovery."""
    
    print("\n" + "=" * 80)
    print("TEST 3: V-SHAPE RECOVERY")
    print("=" * 80)
    
    request = TierBasedRequest(
        company_name="Comeback Kids LLC",
        q1_arr=2000000,
        q2_arr=1000000,
        q3_arr=1200000,
        q4_arr=1800000,
        headcount=75,
        sector="Infrastructure & Network"
    )
    
    result = perform_tier_based_forecast(request)
    
    if result['success']:
        print(f"\n✅ Success!")
        print(f"Company: {result['company_name']}")
        print(f"Prediction Method: {result['prediction_method']}")
        
        print(f"\n📊 Trend Analysis:")
        trend = result['trend_analysis']
        print(f"  Type: {trend['trend_type']}")
        print(f"  Recent Momentum: {trend['metrics']['recent_momentum']*100:+.1f}%")
        print(f"  Message: {trend['user_message']}")
        
        if 'gpt_analysis' in result:
            print(f"\n💡 GPT Reasoning: {result['gpt_analysis']['reasoning']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_declining_company()
    test_growing_company()
    test_v_shape_recovery()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)

