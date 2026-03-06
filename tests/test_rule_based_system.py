#!/usr/bin/env python3
"""
Test the Rule-Based Health Assessment System
=============================================

Tests the new rule-based predictor with various company scenarios.
"""

from rule_based_health_predictor import RuleBasedHealthPredictor
from trend_detector import TrendDetector
from hybrid_prediction_system import HybridPredictionSystem


def test_rule_based_predictor_direct():
    """Test the rule-based predictor directly."""
    
    print("=" * 80)
    print("TEST 1: Direct Rule-Based Predictor Tests")
    print("=" * 80)
    
    predictor = RuleBasedHealthPredictor()
    detector = TrendDetector()
    
    test_cases = [
        {
            'name': 'HIGH HEALTH - Strong Growth Company',
            'q1': 1000000,
            'q2': 1400000,
            'q3': 2000000,
            'q4': 2800000,
            'sector': 'Data & Analytics',
            'headcount': 100,
            'tier2': {
                'gross_margin': 80,
                'sales_marketing': 1200000,
                'cash_burn': -800000,
                'churn_rate': 0.03,  # 3% churn
                'expansion_rate': 0.25,  # 25% expansion
                'customers': 500,
                'runway_months': 24
            }
        },
        {
            'name': 'LOW HEALTH - Declining Company',
            'q1': 2000000,
            'q2': 1500000,
            'q3': 1000000,
            'q4': 500000,
            'sector': 'Data & Analytics',
            'headcount': 50,
            'tier2': {
                'gross_margin': 70,
                'sales_marketing': 200000,
                'cash_burn': -300000,
                'churn_rate': 0.15,  # 15% churn (high!)
                'expansion_rate': 0.02,  # 2% expansion (low!)
                'customers': 100,
                'runway_months': 8
            }
        },
        {
            'name': 'MODERATE HEALTH - Steady Growth',
            'q1': 2000000,
            'q2': 2200000,
            'q3': 2400000,
            'q4': 2600000,
            'sector': 'Cyber Security',
            'headcount': 75,
            'tier2': None  # Will use defaults
        }
    ]
    
    for test in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST CASE: {test['name']}")
        print("=" * 80)
        
        print(f"\nInput Data:")
        print(f"  Q1 2023: ${test['q1']:,}")
        print(f"  Q2 2023: ${test['q2']:,}")
        print(f"  Q3 2023: ${test['q3']:,}")
        print(f"  Q4 2023: ${test['q4']:,}")
        print(f"  Sector: {test['sector']}")
        print(f"  Headcount: {test['headcount']}")
        
        if test['tier2']:
            print(f"\nTier 2 Metrics:")
            for key, value in test['tier2'].items():
                if key == 'churn_rate' or key == 'expansion_rate':
                    print(f"  {key}: {value*100:.1f}%")
                elif key == 'runway_months':
                    print(f"  {key}: {value} months")
                else:
                    print(f"  {key}: ${value:,}" if isinstance(value, (int, float)) and value > 1000 else f"  {key}: {value}")
        
        # Detect trend
        trend_analysis = detector.detect_trend(test['q1'], test['q2'], test['q3'], test['q4'])
        print(f"\n📊 Trend Analysis:")
        print(f"  Type: {trend_analysis['trend_type']}")
        print(f"  Use Rule-Based: {trend_analysis['use_gpt']}")
        
        # Get prediction
        result = predictor.predict_arr(
            q1=test['q1'],
            q2=test['q2'],
            q3=test['q3'],
            q4=test['q4'],
            sector=test['sector'],
            headcount=test['headcount'],
            trend_analysis=trend_analysis,
            tier2_data=test.get('tier2')
        )
        
        # Display health assessment
        print(f"\n🏥 Health Assessment:")
        print(predictor.get_health_summary(result['health_assessment'], result['health_metrics']))
        
        # Display predictions
        print(f"\n📈 Predictions:")
        print("-" * 80)
        for pred in result['predictions']:
            print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} "
                  f"(YoY: {pred['YoY_Growth_Percent']:+.1f}%, "
                  f"QoQ: {pred['QoQ_Growth_Percent']:+.1f}%)")
        
        print(f"\n💡 Reasoning: {result['reasoning']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📝 Key Assumption: {result['key_assumption']}")


def test_hybrid_system_integration():
    """Test the hybrid system with rule-based predictor integrated."""
    
    print("\n\n" + "=" * 80)
    print("TEST 2: Hybrid System Integration Test")
    print("=" * 80)
    
    system = HybridPredictionSystem()
    
    # Test declining company (should use rule-based)
    print("\n" + "=" * 80)
    print("TEST: Declining Company (Should Use Rule-Based)")
    print("=" * 80)
    
    declining_data = {
        'q1_arr': 2000000,
        'q2_arr': 1500000,
        'q3_arr': 1000000,
        'q4_arr': 500000,
        'headcount': 50,
        'sector': 'Data & Analytics'
    }
    
    tier2_data = {
        'gross_margin': 70,
        'sales_marketing': 200000,
        'cash_burn': -300000,
        'churn_rate': 0.15,
        'expansion_rate': 0.02,
        'customers': 100,
        'runway_months': 8
    }
    
    predictions, metadata = system.predict_with_hybrid(declining_data, tier2_data)
    
    print(f"\n✅ Prediction Method: {metadata['prediction_method']}")
    print(f"✅ Trend Type: {metadata['trend_analysis']['trend_type']}")
    
    if 'health_tier' in metadata:
        print(f"\n🏥 Health Tier: {metadata['health_tier']}")
        print(f"📊 Health Score: {metadata['health_assessment']['score']}/100")
        print(f"💡 Reasoning: {metadata['reasoning']}")
        print(f"🎯 Confidence: {metadata['confidence']}")
        
        print(f"\n📈 Predictions:")
        for pred in predictions:
            print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} "
                  f"(YoY: {pred['YoY_Growth_Percent']:+.1f}%, "
                  f"QoQ: {pred['QoQ_Growth_Percent']:+.1f}%)")
    
    # Test growing company (should use ML model)
    print("\n\n" + "=" * 80)
    print("TEST: Growing Company (Should Use ML Model)")
    print("=" * 80)
    
    growing_data = {
        'q1_arr': 1000000,
        'q2_arr': 1400000,
        'q3_arr': 2000000,
        'q4_arr': 2800000,
        'headcount': 100,
        'sector': 'Data & Analytics'
    }
    
    predictions, metadata = system.predict_with_hybrid(growing_data)
    
    print(f"\n✅ Prediction Method: {metadata['prediction_method']}")
    print(f"✅ Trend Type: {metadata['trend_analysis']['trend_type']}")
    
    if 'health_tier' in metadata:
        print(f"\n🏥 Health Tier: {metadata['health_tier']}")
    else:
        print(f"\n📊 Using ML Model (standard growth pattern)")


if __name__ == "__main__":
    # Test 1: Direct rule-based predictor
    test_rule_based_predictor_direct()
    
    # Test 2: Hybrid system integration
    test_hybrid_system_integration()
    
    print("\n\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)

