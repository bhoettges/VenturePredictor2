#!/usr/bin/env python3
"""
Test the specific case reported by user
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from hybrid_prediction_system import HybridPredictionSystem
from trend_detector import TrendDetector

# User's specific values
q1 = 583408
q2 = 604506
q3 = 765056
q4 = 583407

print("=" * 80)
print("TESTING USER'S SPECIFIC CASE")
print("=" * 80)

# Calculate what this looks like
print(f"\nUser Input:")
print(f"  Q1 2023: ${q1:,}")
print(f"  Q2 2023: ${q2:,}")
print(f"  Q3 2023: ${q3:,}")
print(f"  Q4 2023: ${q4:,}")

# Calculate growth rates
qoq1 = ((q2 - q1) / q1) * 100
qoq2 = ((q3 - q2) / q2) * 100
qoq3 = ((q4 - q3) / q3) * 100
total = ((q4 - q1) / q1) * 100

print(f"\nGrowth Analysis:")
print(f"  Q1→Q2: {qoq1:+.1f}%")
print(f"  Q2→Q3: {qoq2:+.1f}%")
print(f"  Q3→Q4: {qoq3:+.1f}% (BIG DROP!)")
print(f"  Q1→Q4: {total:+.1f}% (essentially FLAT)")

# Test trend detection
detector = TrendDetector()
trend = detector.detect_trend(q1, q2, q3, q4)

print(f"\n📊 TREND DETECTION:")
print(f"  Type: {trend['trend_type']}")
print(f"  Use GPT: {trend['use_gpt']}")
print(f"  Confidence: {trend['confidence']}")
print(f"  Message: {trend['user_message']}")
print(f"  Reason: {trend['reason']}")

print(f"\n  Metrics:")
print(f"    Volatility: {trend['metrics']['volatility']:.3f}")
print(f"    Recent Momentum: {trend['metrics']['recent_momentum']*100:+.1f}%")
print(f"    Acceleration: {trend['metrics']['acceleration']}")

# Test hybrid system
print("\n" + "=" * 80)
print("HYBRID SYSTEM PREDICTION")
print("=" * 80)

system = HybridPredictionSystem()

tier1_data = {
    'q1_arr': q1,
    'q2_arr': q2,
    'q3_arr': q3,
    'q4_arr': q4,
    'headcount': 50,
    'sector': 'Data & Analytics'
}

predictions, metadata = system.predict_with_hybrid(tier1_data)

print(f"\n📈 PREDICTIONS:")
for pred in predictions:
    print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} (YoY: {pred['YoY_Growth_Percent']:+.1f}%, QoQ: {pred['QoQ_Growth_Percent']:+.1f}%)")

print(f"\n🤔 ANALYSIS:")
print(f"  Prediction Method: {metadata['prediction_method']}")

# Check if predictions are too similar to input
print(f"\n⚠️  SIMILARITY CHECK:")
input_values = [q1, q2, q3, q4]
predicted_values = [p['ARR'] for p in predictions]

for i, (input_val, pred_val) in enumerate(zip(input_values, predicted_values)):
    diff_pct = abs((pred_val - input_val) / input_val) * 100
    if diff_pct < 5:
        print(f"  Q{i+1} 2024: TOO SIMILAR to Q{i+1} 2023! (only {diff_pct:.1f}% difference)")
    else:
        print(f"  Q{i+1} 2024: OK ({diff_pct:.1f}% difference from Q{i+1} 2023)")

# What SHOULD happen
print(f"\n💡 WHAT SHOULD HAPPEN:")
print(f"  Pattern: Volatile (up +3.6%, spike +26.5%, crash -23.8%)")
print(f"  Recent Momentum: -23.8% (declining)")
print(f"  Expected: GPT should recognize volatility/recent decline")
print(f"  Expected: Predictions should NOT just repeat 2023 values")

