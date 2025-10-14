#!/usr/bin/env python3
"""
Analyze different trend patterns to build robust trend detection
"""

def analyze_trend_patterns():
    """Show different trend scenarios that need different handling."""
    
    scenarios = [
        {
            "name": "Consistent Decline",
            "q1": 2000000, "q2": 1500000, "q3": 1000000, "q4": 500000,
            "pattern": "Each quarter worse than the last"
        },
        {
            "name": "V-Shape Recovery",
            "q1": 2000000, "q2": 1000000, "q3": 1200000, "q4": 1800000,
            "pattern": "Declined then recovered strongly"
        },
        {
            "name": "Late Stage Decline",
            "q1": 2000000, "q2": 2200000, "q3": 2100000, "q4": 1500000,
            "pattern": "Was growing, then sharp drop in Q4"
        },
        {
            "name": "Plateau After Growth",
            "q1": 1000000, "q2": 1500000, "q3": 1800000, "q4": 1850000,
            "pattern": "Strong growth then slowing down"
        },
        {
            "name": "Volatile/Seasonal",
            "q1": 1000000, "q2": 1500000, "q3": 1200000, "q4": 1600000,
            "pattern": "Up and down, but net positive"
        },
        {
            "name": "Flat/Stagnant",
            "q1": 1000000, "q2": 1050000, "q3": 980000, "q4": 1020000,
            "pattern": "Minimal growth, hovering around same level"
        }
    ]
    
    print("=" * 100)
    print("TREND PATTERN ANALYSIS - Why Q1→Q4 Alone Isn't Enough")
    print("=" * 100)
    
    for scenario in scenarios:
        q1, q2, q3, q4 = scenario['q1'], scenario['q2'], scenario['q3'], scenario['q4']
        
        # Simple Q1→Q4
        simple_growth = ((q4 - q1) / q1) * 100
        
        # QoQ changes
        qoq1 = ((q2 - q1) / q1) * 100
        qoq2 = ((q3 - q2) / q2) * 100
        qoq3 = ((q4 - q3) / q3) * 100
        
        # Recent momentum (Q3→Q4)
        recent_momentum = qoq3
        
        # Average QoQ
        avg_qoq = (qoq1 + qoq2 + qoq3) / 3
        
        # Trend consistency (are all QoQ in same direction?)
        all_growing = qoq1 > 0 and qoq2 > 0 and qoq3 > 0
        all_declining = qoq1 < 0 and qoq2 < 0 and qoq3 < 0
        
        # Acceleration/deceleration
        if qoq3 > qoq2 > qoq1:
            acceleration = "Accelerating Growth"
        elif qoq3 < qoq2 < qoq1:
            acceleration = "Decelerating"
        else:
            acceleration = "Irregular"
        
        print(f"\n{scenario['name'].upper()}")
        print(f"  Pattern: {scenario['pattern']}")
        print(f"  ARR: Q1=${q1:,} → Q2=${q2:,} → Q3=${q3:,} → Q4=${q4:,}")
        print(f"\n  Simple Q1→Q4: {simple_growth:+.1f}%")
        print(f"  QoQ Growth: {qoq1:+.1f}% → {qoq2:+.1f}% → {qoq3:+.1f}%")
        print(f"  Recent Momentum (Q3→Q4): {recent_momentum:+.1f}%")
        print(f"  Average QoQ: {avg_qoq:+.1f}%")
        print(f"  Consistency: {'✅ All Growing' if all_growing else '❌ All Declining' if all_declining else '⚠️ Mixed'}")
        print(f"  Trend: {acceleration}")
        
        # What prediction approach makes sense?
        print(f"\n  🎯 PREDICTION STRATEGY:")
        if all_declining and recent_momentum < -10:
            print(f"     → GPT (ML model won't handle consistent decline)")
        elif all_growing and simple_growth > 30:
            print(f"     → ML Model (trained for growth scenarios)")
        elif abs(recent_momentum) > 20 and simple_growth * recent_momentum < 0:
            print(f"     → GPT (recent trend reversal, ML won't capture)")
        elif abs(avg_qoq) < 5:
            print(f"     → GPT (flat/stagnant, ML expects growth)")
        else:
            print(f"     → ML Model (standard growth pattern)")
        
        print(f"  " + "-" * 80)

def proposed_trend_detection():
    """Propose a multi-factor trend detection system."""
    
    print("\n" + "=" * 100)
    print("PROPOSED MULTI-FACTOR TREND DETECTION")
    print("=" * 100)
    
    print("""
    def detect_company_trend(q1, q2, q3, q4):
        # Calculate all metrics
        simple_growth = (q4 - q1) / q1
        qoq1 = (q2 - q1) / q1
        qoq2 = (q3 - q2) / q2
        qoq3 = (q4 - q3) / q3
        avg_qoq = (qoq1 + qoq2 + qoq3) / 3
        
        # Recent momentum (weighted toward Q4)
        recent_momentum = qoq3
        
        # Consistency check
        all_positive = qoq1 > 0 and qoq2 > 0 and qoq3 > 0
        all_negative = qoq1 < 0 and qoq2 < 0 and qoq3 < 0
        
        # Volatility (standard deviation of QoQ growth)
        import numpy as np
        volatility = np.std([qoq1, qoq2, qoq3])
        
        # CLASSIFICATION LOGIC
        
        # 1. CONSISTENT DECLINE (use GPT)
        if all_negative or (simple_growth < -0.15 and recent_momentum < -0.10):
            return {
                "trend": "CONSISTENT_DECLINE",
                "use_gpt": True,
                "confidence": "high",
                "reason": "Sustained negative trend"
            }
        
        # 2. RECENT REVERSAL (use GPT - ML won't see this)
        if (simple_growth > 0 and recent_momentum < -0.15) or \
           (simple_growth < 0 and recent_momentum > 0.15):
            return {
                "trend": "TREND_REVERSAL",
                "use_gpt": True,
                "confidence": "medium",
                "reason": "Recent momentum contradicts overall trend"
            }
        
        # 3. FLAT/STAGNANT (use GPT - ML expects growth)
        if abs(simple_growth) < 0.10 and abs(avg_qoq) < 0.05:
            return {
                "trend": "FLAT_STAGNANT",
                "use_gpt": True,
                "confidence": "high",
                "reason": "Minimal growth, ML trained on high-growth companies"
            }
        
        # 4. HIGHLY VOLATILE (use GPT for reasoning)
        if volatility > 0.25:  # High variance in QoQ growth
            return {
                "trend": "VOLATILE_IRREGULAR",
                "use_gpt": True,
                "confidence": "medium",
                "reason": "Irregular pattern requires contextual analysis"
            }
        
        # 5. CONSISTENT GROWTH (use ML - this is what it's trained for!)
        if all_positive and simple_growth > 0.20:
            return {
                "trend": "CONSISTENT_GROWTH",
                "use_gpt": False,
                "confidence": "high",
                "reason": "Standard growth pattern, ML model excels here"
            }
        
        # 6. MODERATE GROWTH (use ML, but with caution)
        if simple_growth > 0:
            return {
                "trend": "MODERATE_GROWTH",
                "use_gpt": False,
                "confidence": "medium",
                "reason": "Positive trend, ML model suitable"
            }
        
        # 7. DEFAULT - Use GPT for safety
        return {
            "trend": "UNCLEAR",
            "use_gpt": True,
            "confidence": "low",
            "reason": "Pattern doesn't fit known categories"
        }
    """)
    
    print("\n" + "=" * 100)
    print("KEY FACTORS:")
    print("=" * 100)
    print("""
    1. ✅ Q1→Q4 Overall Growth (simple_growth)
    2. ✅ Individual QoQ Changes (qoq1, qoq2, qoq3)
    3. ✅ Recent Momentum (Q3→Q4, most important for next prediction)
    4. ✅ Consistency (all same direction vs mixed)
    5. ✅ Volatility (how erratic is the pattern)
    6. ✅ Acceleration/Deceleration (speeding up or slowing down)
    
    ROUTING DECISION:
    - Use ML Model: Consistent growth patterns (93% of training data)
    - Use GPT: Declines, reversals, flat, volatile (edge cases)
    """)

if __name__ == "__main__":
    analyze_trend_patterns()
    proposed_trend_detection()

