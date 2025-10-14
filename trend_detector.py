#!/usr/bin/env python3
"""
Multi-Factor Trend Detection Module
====================================

Analyzes company trajectory using multiple factors to determine
the best prediction method (ML model vs GPT).
"""

import numpy as np
from typing import Dict, List

class TrendDetector:
    """Detects company trends using multi-factor analysis."""
    
    def __init__(self):
        # Thresholds for classification
        self.DECLINE_THRESHOLD = -0.15  # 15% decline
        self.GROWTH_THRESHOLD = 0.20    # 20% growth
        self.FLAT_THRESHOLD = 0.10      # 10% for flat classification
        self.MOMENTUM_THRESHOLD = 0.15  # 15% for reversal detection
        self.VOLATILITY_THRESHOLD = 0.25  # High variance threshold
    
    def detect_trend(self, q1: float, q2: float, q3: float, q4: float) -> Dict:
        """
        Analyze trend using multiple factors.
        
        Returns:
            dict with trend classification, routing decision, and metrics
        """
        # Calculate all growth metrics
        simple_growth = (q4 - q1) / q1 if q1 > 0 else 0
        qoq1 = (q2 - q1) / q1 if q1 > 0 else 0
        qoq2 = (q3 - q2) / q2 if q2 > 0 else 0
        qoq3 = (q4 - q3) / q3 if q3 > 0 else 0
        
        avg_qoq = (qoq1 + qoq2 + qoq3) / 3
        recent_momentum = qoq3  # Q3→Q4 most important for next prediction
        
        # Consistency checks
        all_positive = qoq1 > 0 and qoq2 > 0 and qoq3 > 0
        all_negative = qoq1 < 0 and qoq2 < 0 and qoq3 < 0
        
        # Volatility (standard deviation of QoQ growth)
        volatility = np.std([qoq1, qoq2, qoq3])
        
        # Acceleration pattern
        if qoq3 > qoq2 > qoq1:
            acceleration = "accelerating"
        elif qoq3 < qoq2 < qoq1:
            acceleration = "decelerating"
        else:
            acceleration = "irregular"
        
        # Build metrics dict
        metrics = {
            "q1_arr": q1,
            "q2_arr": q2,
            "q3_arr": q3,
            "q4_arr": q4,
            "simple_growth": simple_growth,
            "qoq_growth": [qoq1, qoq2, qoq3],
            "avg_qoq": avg_qoq,
            "recent_momentum": recent_momentum,
            "volatility": volatility,
            "acceleration": acceleration,
            "all_positive": all_positive,
            "all_negative": all_negative
        }
        
        # CLASSIFICATION LOGIC
        
        # 1. CONSISTENT DECLINE (use GPT)
        if all_negative or (simple_growth < self.DECLINE_THRESHOLD and recent_momentum < -0.10):
            return {
                "trend_type": "CONSISTENT_DECLINE",
                "use_gpt": True,
                "confidence": "high",
                "reason": "Sustained negative trend - ML model trained on growth",
                "metrics": metrics,
                "user_message": "⚠️ Company showing consistent decline. Using advanced analysis."
            }
        
        # 2. RECENT REVERSAL (use GPT - ML won't capture this)
        if (simple_growth > 0 and recent_momentum < -self.MOMENTUM_THRESHOLD) or \
           (simple_growth < 0 and recent_momentum > self.MOMENTUM_THRESHOLD):
            return {
                "trend_type": "TREND_REVERSAL",
                "use_gpt": True,
                "confidence": "medium",
                "reason": "Recent momentum contradicts overall trend - requires contextual analysis",
                "metrics": metrics,
                "user_message": "📊 Trend reversal detected. Using contextual analysis."
            }
        
        # 3. FLAT/STAGNANT (use GPT - ML expects growth)
        if abs(simple_growth) < self.FLAT_THRESHOLD and abs(avg_qoq) < 0.05:
            return {
                "trend_type": "FLAT_STAGNANT",
                "use_gpt": True,
                "confidence": "high",
                "reason": "Minimal growth - ML model trained on high-growth companies",
                "metrics": metrics,
                "user_message": "📉 Flat growth pattern detected. Using specialized analysis."
            }
        
        # 4. HIGHLY VOLATILE (use GPT for contextual reasoning)
        if volatility > self.VOLATILITY_THRESHOLD:
            return {
                "trend_type": "VOLATILE_IRREGULAR",
                "use_gpt": True,
                "confidence": "medium",
                "reason": f"High volatility ({volatility:.2f}) - irregular pattern requires contextual analysis",
                "metrics": metrics,
                "user_message": "⚡ High volatility detected. Using contextual analysis."
            }
        
        # 5. CONSISTENT GROWTH (use ML - this is what it's trained for!)
        if all_positive and simple_growth > self.GROWTH_THRESHOLD:
            return {
                "trend_type": "CONSISTENT_GROWTH",
                "use_gpt": False,
                "confidence": "high",
                "reason": "Standard growth pattern - ML model excels here",
                "metrics": metrics,
                "user_message": "📈 Strong growth pattern. Using ML model predictions."
            }
        
        # 6. MODERATE GROWTH (use ML, but with caution)
        if simple_growth > 0:
            return {
                "trend_type": "MODERATE_GROWTH",
                "use_gpt": False,
                "confidence": "medium",
                "reason": "Positive trend - ML model suitable",
                "metrics": metrics,
                "user_message": "📊 Moderate growth detected. Using ML model predictions."
            }
        
        # 7. DEFAULT - Use GPT for safety
        return {
            "trend_type": "UNCLEAR",
            "use_gpt": True,
            "confidence": "low",
            "reason": "Pattern doesn't fit standard categories - using contextual analysis",
            "metrics": metrics,
            "user_message": "🔍 Unclear pattern. Using advanced contextual analysis."
        }
    
    def get_trend_summary(self, trend_result: Dict) -> str:
        """Generate human-readable trend summary."""
        metrics = trend_result["metrics"]
        
        summary = f"""
Trend Analysis:
  Type: {trend_result['trend_type']}
  Confidence: {trend_result['confidence']}
  
  Q1→Q4 Growth: {metrics['simple_growth']*100:+.1f}%
  QoQ Pattern: {metrics['qoq_growth'][0]*100:+.1f}% → {metrics['qoq_growth'][1]*100:+.1f}% → {metrics['qoq_growth'][2]*100:+.1f}%
  Recent Momentum: {metrics['recent_momentum']*100:+.1f}%
  Volatility: {metrics['volatility']:.2f}
  Pattern: {metrics['acceleration']}
  
  Prediction Method: {'GPT (Contextual Analysis)' if trend_result['use_gpt'] else 'ML Model'}
  Reason: {trend_result['reason']}
"""
        return summary

def test_trend_detector():
    """Test the trend detector with various scenarios."""
    
    detector = TrendDetector()
    
    test_cases = [
        {
            "name": "Consistent Decline",
            "data": (2000000, 1500000, 1000000, 500000)
        },
        {
            "name": "V-Shape Recovery",
            "data": (2000000, 1000000, 1200000, 1800000)
        },
        {
            "name": "Strong Growth",
            "data": (1000000, 1400000, 2000000, 2800000)
        },
        {
            "name": "Flat/Stagnant",
            "data": (1000000, 1050000, 980000, 1020000)
        },
        {
            "name": "Late Decline",
            "data": (2000000, 2200000, 2100000, 1500000)
        }
    ]
    
    for test in test_cases:
        print("=" * 80)
        print(f"TEST: {test['name']}")
        print("=" * 80)
        
        result = detector.detect_trend(*test['data'])
        print(result['user_message'])
        print(detector.get_trend_summary(result))

if __name__ == "__main__":
    test_trend_detector()

