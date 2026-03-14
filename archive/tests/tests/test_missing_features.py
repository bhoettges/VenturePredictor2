#!/usr/bin/env python3
"""
Test what happens when users don't provide all Tier 2 features
"""

from rule_based_health_predictor import RuleBasedHealthPredictor
from trend_detector import TrendDetector


def test_missing_features():
    """Test with various levels of missing data."""
    
    print("=" * 80)
    print("TEST: Missing Features Handling")
    print("=" * 80)
    
    predictor = RuleBasedHealthPredictor()
    detector = TrendDetector()
    
    # Base company data
    q1, q2, q3, q4 = 2000000, 1500000, 1000000, 500000
    sector = "Data & Analytics"
    headcount = 50
    
    trend_analysis = detector.detect_trend(q1, q2, q3, q4)
    
    test_cases = [
        {
            'name': 'No Tier 2 Data (All Estimated)',
            'tier2': None
        },
        {
            'name': 'Partial Tier 2 Data (Some Estimated)',
            'tier2': {
                'gross_margin': 75,
                'churn_rate': 0.10,
                # Missing: sales_marketing, cash_burn, expansion_rate, customers, runway_months
            }
        },
        {
            'name': 'Complete Tier 2 Data (All Provided)',
            'tier2': {
                'gross_margin': 70,
                'sales_marketing': 200000,
                'cash_burn': -300000,
                'churn_rate': 0.15,
                'expansion_rate': 0.02,
                'customers': 100,
                'runway_months': 8
            }
        }
    ]
    
    for test in test_cases:
        print("\n" + "=" * 80)
        print(f"TEST: {test['name']}")
        print("=" * 80)
        
        result = predictor.predict_arr(
            q1=q1, q2=q2, q3=q3, q4=q4,
            sector=sector,
            headcount=headcount,
            trend_analysis=trend_analysis,
            tier2_data=test['tier2']
        )
        
        metrics = result['health_metrics']
        assessment = result['health_assessment']
        
        print(f"\n📊 Calculated Metrics:")
        print(f"  ARR Growth: {metrics['arr_growth_yoy_percent']:.1f}%")
        print(f"  NRR: {metrics['nrr']:.1f}%")
        print(f"  CAC Payback: {metrics['cac_payback_months']:.0f} months")
        print(f"  Rule of 40: {metrics['rule_of_40']:.1f}%")
        print(f"  Runway: {metrics['runway_months']:.0f} months")
        
        print(f"\n🏥 Health Assessment:")
        print(f"  Tier: {assessment['tier']}")
        print(f"  Score: {assessment['score']}/100")
        
        # Show which metrics were estimated (from the predictor's response)
        estimated_metrics = result.get('estimated_metrics', [])
        
        if estimated_metrics:
            print(f"\n⚠️  Estimated Metrics (based on ARR trends):")
            for metric in estimated_metrics:
                print(f"  • {metric}")
        else:
            print(f"\n✅ All metrics calculated from provided data")
        
        # Show confidence impact
        if estimated_metrics:
            print(f"\n📊 Note: Predictions use estimated metrics, which may reduce accuracy.")
            print(f"   Providing Tier 2 data (churn, expansion, S&M, etc.) will improve precision.")
        
        print(f"\n📈 Predictions:")
        for pred in result['predictions'][:2]:  # Show first 2 quarters
            print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f}")


if __name__ == "__main__":
    test_missing_features()

