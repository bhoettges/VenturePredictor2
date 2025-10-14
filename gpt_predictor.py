#!/usr/bin/env python3
"""
GPT-Based Prediction System for Edge Cases
==========================================

Uses OpenAI GPT to make contextual predictions for companies
that don't fit standard growth patterns.
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

class GPTPredictor:
    """Uses GPT for contextual financial predictions."""
    
    def __init__(self):
        """Initialize GPT predictor with OpenAI."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,  # Lower temperature for more consistent predictions
            openai_api_key=api_key
        )
    
    def predict_arr(
        self,
        q1: float,
        q2: float,
        q3: float,
        q4: float,
        sector: str,
        headcount: int,
        trend_analysis: Dict
    ) -> Dict:
        """
        Use GPT to predict ARR for next 4 quarters with reasoning.
        
        Args:
            q1-q4: Historical ARR values
            sector: Company sector
            headcount: Number of employees
            trend_analysis: Output from TrendDetector
            
        Returns:
            Dict with predictions and reasoning
        """
        
        # Calculate metrics for context
        metrics = trend_analysis['metrics']
        
        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial forecasting expert specializing in SaaS companies. 
You analyze company trajectories and provide realistic ARR predictions based on recent trends, 
industry patterns, and business fundamentals. Your predictions should be data-driven and conservative."""),
            ("user", """Analyze this SaaS company and predict Q1-Q4 2024 ARR values.

Company Profile:
- Sector: {sector}
- Team Size: {headcount} employees
- ARR Trend: Q1 2023: ${q1:,} → Q2: ${q2:,} → Q3: ${q3:,} → Q4: ${q4:,}

Trend Analysis:
- Pattern Type: {trend_type}
- Q1→Q4 Growth: {simple_growth:+.1%}
- QoQ Changes: {qoq1:+.1%} → {qoq2:+.1%} → {qoq3:+.1%}
- Recent Momentum (Q3→Q4): {recent_momentum:+.1%}
- Volatility: {volatility:.2f}
- Pattern: {acceleration}

Context: {reason}

Based on this {trend_type} pattern, provide realistic ARR predictions for 2024:

1. Consider the ACTUAL trajectory (not just overall growth)
2. Factor in recent momentum (most important indicator)
3. Apply industry recovery/decline patterns
4. Be conservative with reversals
5. Account for sector-specific dynamics

Return ONLY a JSON object with this exact structure (no other text):
{{
    "q1_2024": <number>,
    "q2_2024": <number>,
    "q3_2024": <number>,
    "q4_2024": <number>,
    "reasoning": "<brief explanation of prediction logic>",
    "confidence": "<high/medium/low>",
    "key_assumption": "<main assumption driving prediction>"
}}

Example for a declining company:
{{
    "q1_2024": 450000,
    "q2_2024": 420000,
    "q3_2024": 400000,
    "q4_2024": 390000,
    "reasoning": "Continued decline expected with deceleration as company stabilizes",
    "confidence": "medium",
    "key_assumption": "No turnaround initiatives in place, natural stabilization over time"
}}""")
        ])
        
        # Format the prompt
        formatted_prompt = prompt.format_messages(
            sector=sector,
            headcount=headcount,
            q1=q1, q2=q2, q3=q3, q4=q4,
            trend_type=trend_analysis['trend_type'],
            simple_growth=metrics['simple_growth'],
            qoq1=metrics['qoq_growth'][0],
            qoq2=metrics['qoq_growth'][1],
            qoq3=metrics['qoq_growth'][2],
            recent_momentum=metrics['recent_momentum'],
            volatility=metrics['volatility'],
            acceleration=metrics['acceleration'],
            reason=trend_analysis['reason']
        )
        
        try:
            # Get GPT response
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content.strip()
            
            # Parse JSON response
            # Handle cases where GPT returns markdown code blocks
            if response_text.startswith('```'):
                # Extract JSON from code block
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            prediction_data = json.loads(response_text)
            
            # Validate response structure
            required_fields = ['q1_2024', 'q2_2024', 'q3_2024', 'q4_2024', 'reasoning', 'confidence']
            if not all(field in prediction_data for field in required_fields):
                raise ValueError(f"GPT response missing required fields. Got: {prediction_data.keys()}")
            
            # Format predictions with confidence intervals
            predictions = []
            quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
            arr_values = [
                prediction_data['q1_2024'],
                prediction_data['q2_2024'],
                prediction_data['q3_2024'],
                prediction_data['q4_2024']
            ]
            
            for quarter, arr in zip(quarters, arr_values):
                # Calculate YoY growth (comparing to same quarter last year)
                quarter_idx = quarters.index(quarter)
                base_arr = [q1, q2, q3, q4][quarter_idx]
                yoy_growth = ((arr - base_arr) / base_arr) if base_arr > 0 else 0
                
                # Add confidence intervals (±10%)
                predictions.append({
                    'Quarter': quarter,
                    'ARR': arr,
                    'Pessimistic_ARR': arr * 0.9,
                    'Optimistic_ARR': arr * 1.1,
                    'YoY_Growth': yoy_growth,
                    'YoY_Growth_Percent': yoy_growth * 100,
                    'QoQ_Growth_Percent': 0  # Will be calculated in API layer
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'gpt_reasoning': prediction_data['reasoning'],
                'gpt_confidence': prediction_data['confidence'],
                'gpt_assumption': prediction_data.get('key_assumption', 'N/A'),
                'raw_response': prediction_data
            }
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse GPT response: {e}")
            print(f"Raw response: {response_text}")
            return self._fallback_prediction(q1, q2, q3, q4, metrics)
        
        except Exception as e:
            print(f"GPT prediction error: {e}")
            return self._fallback_prediction(q1, q2, q3, q4, metrics)
    
    def _fallback_prediction(self, q1: float, q2: float, q3: float, q4: float, metrics: Dict) -> Dict:
        """
        Fallback rule-based prediction if GPT fails.
        Uses recent momentum to project forward.
        """
        print("⚠️ Using fallback prediction logic (GPT failed)")
        
        recent_momentum = metrics['recent_momentum']
        avg_qoq = metrics['avg_qoq']
        
        # Project forward using dampened recent momentum
        # (assume trend continues but decelerates)
        dampening_factor = 0.8
        
        predictions = []
        current_arr = q4
        
        for i, quarter in enumerate(['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']):
            # Apply dampened momentum
            growth_rate = recent_momentum * (dampening_factor ** i)
            next_arr = current_arr * (1 + growth_rate)
            
            # Calculate YoY
            base_arr = [q1, q2, q3, q4][i]
            yoy_growth = ((next_arr - base_arr) / base_arr) if base_arr > 0 else 0
            
            predictions.append({
                'Quarter': quarter,
                'ARR': next_arr,
                'Pessimistic_ARR': next_arr * 0.9,
                'Optimistic_ARR': next_arr * 1.1,
                'YoY_Growth': yoy_growth,
                'YoY_Growth_Percent': yoy_growth * 100,
                'QoQ_Growth_Percent': 0
            })
            
            current_arr = next_arr
        
        return {
            'success': True,
            'predictions': predictions,
            'gpt_reasoning': f'Fallback: Projected using recent momentum ({recent_momentum*100:.1f}%) with dampening',
            'gpt_confidence': 'low',
            'gpt_assumption': 'Recent trend continues but decelerates over time',
            'fallback_used': True
        }

def test_gpt_predictor():
    """Test GPT predictor with sample data."""
    
    from trend_detector import TrendDetector
    
    detector = TrendDetector()
    predictor = GPTPredictor()
    
    # Test declining company
    print("=" * 80)
    print("TEST: GPT Prediction for Declining Company")
    print("=" * 80)
    
    q1, q2, q3, q4 = 2000000, 1500000, 1000000, 500000
    trend_analysis = detector.detect_trend(q1, q2, q3, q4)
    
    print(f"\nTrend Type: {trend_analysis['trend_type']}")
    print(f"Use GPT: {trend_analysis['use_gpt']}")
    
    if trend_analysis['use_gpt']:
        result = predictor.predict_arr(
            q1=q1, q2=q2, q3=q3, q4=q4,
            sector="Data & Analytics",
            headcount=50,
            trend_analysis=trend_analysis
        )
        
        print(f"\nGPT Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['Quarter']}: ${pred['ARR']:,.0f} ({pred['YoY_Growth_Percent']:+.1f}% YoY)")
        
        print(f"\nGPT Reasoning: {result['gpt_reasoning']}")
        print(f"Confidence: {result['gpt_confidence']}")
        print(f"Key Assumption: {result['gpt_assumption']}")

if __name__ == "__main__":
    test_gpt_predictor()

