"""
LangChain Tools for Prediction Analysis
Provides tools for the chat to analyze stored predictions
"""

from langchain.tools import Tool
from prediction_memory import prediction_memory
import json
from typing import Dict, List, Any

def analyze_recent_predictions(query: str) -> str:
    """Analyze recent predictions based on user query"""
    try:
        recent_predictions = prediction_memory.get_recent_predictions(5)
        
        if not recent_predictions:
            return "No recent predictions available for analysis. Please make a prediction first by providing your company's quarterly ARR data."
        
        # Get the most recent prediction (user's latest forecast)
        latest_prediction = recent_predictions[0] if recent_predictions else None
        
        # Parse the query to understand what the user wants to know
        query_lower = query.lower()
        
        # Check if user is asking "why" or questioning the prediction
        why_question_keywords = ['why', 'how come', 'explain', 'doesn\'t make sense', 'not', 'never']
        is_why_question = any(keyword in query_lower for keyword in why_question_keywords)
        
        # If asking why/questioning, provide detailed reasoning from latest prediction
        if is_why_question and latest_prediction:
            return explain_latest_prediction(latest_prediction, query)
        
        if "summary" in query_lower or "overview" in query_lower:
            return prediction_memory.get_prediction_summary()
        
        elif "company" in query_lower:
            # Extract company name from query
            words = query.split()
            company_name = None
            for i, word in enumerate(words):
                if word.lower() in ["company", "for", "about"] and i + 1 < len(words):
                    company_name = words[i + 1]
                    break
            
            if company_name:
                prediction = prediction_memory.get_prediction_by_company(company_name)
                if prediction:
                    return format_company_prediction(prediction)
                else:
                    return f"No prediction found for company '{company_name}'. Available companies: {', '.join([p['company_name'] for p in recent_predictions])}"
            else:
                return "Please specify which company you want to analyze. Available companies: " + ", ".join([p['company_name'] for p in recent_predictions])
        
        elif "growth" in query_lower or "performance" in query_lower:
            return analyze_growth_patterns(recent_predictions)
        
        elif "accuracy" in query_lower or "model" in query_lower:
            return analyze_model_performance(recent_predictions)
        
        elif "confidence" in query_lower or "uncertainty" in query_lower:
            return analyze_confidence_intervals(recent_predictions)
        
        else:
            # General analysis
            return general_prediction_analysis(recent_predictions)
    
    except Exception as e:
        return f"Error analyzing predictions: {str(e)}"

def explain_latest_prediction(prediction: Dict[str, Any], query: str) -> str:
    """Explain the latest prediction, addressing user's specific concern"""
    if not prediction['success']:
        return f"Your latest prediction encountered an error: {prediction.get('error', 'Unknown error')}"
    
    # Extract key information
    trend_analysis = prediction.get('trend_analysis', {})
    gpt_analysis = prediction.get('gpt_analysis', {})
    input_data = prediction.get('input_data', {})
    forecasts = prediction.get('forecast', [])
    
    # Get input ARR values
    q1_input = input_data.get('q1_arr', 0)
    q2_input = input_data.get('q2_arr', 0)
    q3_input = input_data.get('q3_arr', 0)
    q4_input = input_data.get('q4_arr', 0)
    
    # Find peak quarter in input
    arr_values = [q1_input, q2_input, q3_input, q4_input]
    max_arr = max(arr_values) if arr_values else 0
    peak_quarter = ['Q1', 'Q2', 'Q3', 'Q4'][arr_values.index(max_arr)] if arr_values else 'Unknown'
    
    # Check if predictions ever reach peak
    predicted_arrs = [f.get('predicted_arr', 0) for f in forecasts]
    max_predicted = max(predicted_arrs) if predicted_arrs else 0
    
    result = f"📊 **Understanding Your Prediction**\n\n"
    
    # Address the specific concern
    if max_arr > max_predicted:
        result += f"**Your Question:** You're right to notice that predictions don't reach your Q{peak_quarter[-1]} peak of ${max_arr:,.0f}!\n\n"
        result += f"**Why This Happens:**\n"
        
        if trend_analysis:
            trend_type = trend_analysis.get('trend_type', 'Unknown')
            recent_momentum = trend_analysis.get('metrics', {}).get('recent_momentum', 0) * 100
            
            result += f"1. **Trend Detected:** {trend_type}\n"
            result += f"2. **Recent Momentum:** {recent_momentum:+.1f}% (Q3→Q4)\n"
            
            if recent_momentum < -15:
                result += f"3. **Recent Crash:** Your company dropped {abs(recent_momentum):.1f}% from Q3 to Q4\n"
                result += f"4. **Prediction Logic:** After a significant drop, the model projects continued decline or slow recovery, not an immediate return to peak\n\n"
            
            if gpt_analysis:
                result += f"**GPT Reasoning:**\n{gpt_analysis.get('reasoning', 'N/A')}\n\n"
                result += f"**Key Assumption:**\n{gpt_analysis.get('key_assumption', 'N/A')}\n\n"
        
        result += f"**The Numbers:**\n"
        result += f"  • Your Q{peak_quarter[-1]} 2023 Peak: ${max_arr:,.0f}\n"
        result += f"  • Your Q4 2023 (Current): ${q4_input:,.0f}\n"
        result += f"  • Highest 2024 Prediction: ${max_predicted:,.0f}\n"
        result += f"  • Gap: ${max_arr - max_predicted:,.0f} ({((max_arr - max_predicted)/max_arr*100):.1f}% below peak)\n\n"
        
        result += f"**What This Means:**\n"
        result += f"The model sees your recent decline and projects recovery will take time. "
        result += f"To reach your Q{peak_quarter[-1]} peak again, you'd likely need operational changes or market improvements.\n\n"
        result += f"💡 **Tip:** If you believe you'll recover faster, this is where business judgment overrides the model!"
    
    else:
        result += f"**Your Input:** Q1 ${q1_input:,.0f}, Q2 ${q2_input:,.0f}, Q3 ${q3_input:,.0f}, Q4 ${q4_input:,.0f}\n\n"
        
        if trend_analysis:
            result += f"**Trend Analysis:**\n"
            result += f"  • Type: {trend_analysis.get('trend_type', 'Unknown')}\n"
            result += f"  • Confidence: {trend_analysis.get('confidence', 'Unknown')}\n"
            result += f"  • Reason: {trend_analysis.get('reason', 'N/A')}\n\n"
        
        if gpt_analysis:
            result += f"**Prediction Reasoning:**\n{gpt_analysis.get('reasoning', 'Standard ML model prediction')}\n\n"
        
        result += f"**2024 Forecast:**\n"
        for forecast in forecasts[:2]:
            quarter = forecast.get('quarter', 'Unknown')
            arr = forecast.get('predicted_arr', 0)
            yoy = forecast.get('yoy_growth_percent', 0)
            result += f"  • {quarter}: ${arr:,.0f} ({yoy:+.1f}% YoY)\n"
    
    return result

def format_company_prediction(prediction: Dict[str, Any]) -> str:
    """Format a specific company's prediction for display"""
    if not prediction['success']:
        return f"❌ **{prediction['company_name']}** prediction failed: {prediction.get('error', 'Unknown error')}"
    
    insights = prediction.get('insights', {})
    forecasts = prediction.get('predictions', [])
    
    result = f"📊 **{prediction['company_name']}** Prediction Analysis\n"
    result += f"**Date:** {prediction['timestamp']}\n"
    result += f"**Model:** {prediction['model_used']}\n\n"
    
    # Key metrics
    current_arr = insights.get('current_arr', 0)
    final_arr = insights.get('predicted_final_arr', 0)
    growth = insights.get('total_growth_percent', 0)
    
    result += f"**Growth Forecast:** {growth:.1f}% YoY\n"
    result += f"**ARR Trajectory:** ${current_arr:,.0f} → ${final_arr:,.0f}\n\n"
    
    # Quarterly breakdown
    if forecasts:
        result += "**Quarterly Forecast:**\n"
        for forecast in forecasts:
            quarter = forecast.get('quarter', 'Unknown')
            arr = forecast.get('predicted_arr', 0)
            pessimistic = forecast.get('pessimistic_arr', 0)
            optimistic = forecast.get('optimistic_arr', 0)
            yoy_growth = forecast.get('yoy_growth_percent', 0)
            
            result += f"  • {quarter}: ${arr:,.0f} ({yoy_growth:.1f}% YoY)\n"
            result += f"    Range: ${pessimistic:,.0f} - ${optimistic:,.0f}\n"
    
    return result

def analyze_growth_patterns(predictions: List[Dict[str, Any]]) -> str:
    """Analyze growth patterns across predictions"""
    successful_predictions = [p for p in predictions if p['success']]
    
    if not successful_predictions:
        return "No successful predictions available for growth analysis."
    
    growth_rates = []
    for pred in successful_predictions:
        insights = pred.get('insights', {})
        growth = insights.get('total_growth_percent', 0)
        if growth > 0:
            growth_rates.append(growth)
    
    if not growth_rates:
        return "No growth data available for analysis."
    
    avg_growth = sum(growth_rates) / len(growth_rates)
    max_growth = max(growth_rates)
    min_growth = min(growth_rates)
    
    result = f"📈 **Growth Pattern Analysis** ({len(growth_rates)} predictions)\n\n"
    result += f"**Average Growth:** {avg_growth:.1f}% YoY\n"
    result += f"**Growth Range:** {min_growth:.1f}% - {max_growth:.1f}%\n\n"
    
    # Growth categories
    high_growth = [g for g in growth_rates if g > 100]
    moderate_growth = [g for g in growth_rates if 20 <= g <= 100]
    low_growth = [g for g in growth_rates if g < 20]
    
    result += f"**Growth Distribution:**\n"
    result += f"  • High Growth (>100%): {len(high_growth)} companies\n"
    result += f"  • Moderate Growth (20-100%): {len(moderate_growth)} companies\n"
    result += f"  • Low Growth (<20%): {len(low_growth)} companies\n"
    
    return result

def analyze_model_performance(predictions: List[Dict[str, Any]]) -> str:
    """Analyze model performance across predictions"""
    successful_predictions = [p for p in predictions if p['success']]
    failed_predictions = [p for p in predictions if not p['success']]
    
    result = f"🎯 **Model Performance Analysis** ({len(predictions)} total predictions)\n\n"
    result += f"**Success Rate:** {len(successful_predictions)}/{len(predictions)} ({len(successful_predictions)/len(predictions)*100:.1f}%)\n\n"
    
    if successful_predictions:
        # Model accuracy info
        model_accuracies = []
        for pred in successful_predictions:
            insights = pred.get('insights', {})
            accuracy = insights.get('model_accuracy', '')
            if 'R²' in accuracy:
                # Extract R² value
                try:
                    r2_str = accuracy.split('R² = ')[1].split(' ')[0]
                    r2_value = float(r2_str)
                    model_accuracies.append(r2_value)
                except:
                    pass
        
        if model_accuracies:
            avg_accuracy = sum(model_accuracies) / len(model_accuracies)
            result += f"**Average Model Accuracy:** R² = {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)\n"
        
        result += f"**Models Used:** {', '.join(set(p['model_used'] for p in successful_predictions))}\n"
    
    if failed_predictions:
        result += f"\n**Failed Predictions:** {len(failed_predictions)}\n"
        for pred in failed_predictions[:3]:  # Show first 3 failures
            result += f"  • {pred['company_name']}: {pred.get('error', 'Unknown error')}\n"
    
    return result

def analyze_confidence_intervals(predictions: List[Dict[str, Any]]) -> str:
    """Analyze confidence intervals across predictions"""
    successful_predictions = [p for p in predictions if p['success']]
    
    if not successful_predictions:
        return "No successful predictions available for confidence analysis."
    
    result = f"📊 **Confidence Interval Analysis** ({len(successful_predictions)} predictions)\n\n"
    
    total_forecasts = 0
    confidence_ranges = []
    
    for pred in successful_predictions:
        forecasts = pred.get('predictions', [])
        for forecast in forecasts:
            arr = forecast.get('predicted_arr', 0)
            pessimistic = forecast.get('pessimistic_arr', 0)
            optimistic = forecast.get('optimistic_arr', 0)
            
            if arr > 0:
                # Calculate confidence range as percentage
                lower_range = (arr - pessimistic) / arr * 100
                upper_range = (optimistic - arr) / arr * 100
                avg_range = (lower_range + upper_range) / 2
                confidence_ranges.append(avg_range)
                total_forecasts += 1
    
    if confidence_ranges:
        avg_confidence_range = sum(confidence_ranges) / len(confidence_ranges)
        result += f"**Average Confidence Range:** ±{avg_confidence_range:.1f}%\n"
        result += f"**Total Forecasts Analyzed:** {total_forecasts}\n\n"
        
        # Confidence distribution
        tight_confidence = [r for r in confidence_ranges if r <= 10]
        moderate_confidence = [r for r in confidence_ranges if 10 < r <= 20]
        wide_confidence = [r for r in confidence_ranges if r > 20]
        
        result += f"**Confidence Distribution:**\n"
        result += f"  • Tight (±≤10%): {len(tight_confidence)} forecasts\n"
        result += f"  • Moderate (±10-20%): {len(moderate_confidence)} forecasts\n"
        result += f"  • Wide (±>20%): {len(wide_confidence)} forecasts\n"
    
    return result

def general_prediction_analysis(predictions: List[Dict[str, Any]]) -> str:
    """General analysis of recent predictions"""
    if not predictions:
        return "No predictions available for analysis."
    
    successful = [p for p in predictions if p['success']]
    failed = [p for p in predictions if not p['success']]
    
    result = f"📋 **Recent Predictions Overview** ({len(predictions)} total)\n\n"
    result += f"**Success Rate:** {len(successful)}/{len(predictions)} ({len(successful)/len(predictions)*100:.1f}%)\n\n"
    
    if successful:
        result += "**Recent Successful Predictions:**\n"
        for pred in successful[:3]:
            insights = pred.get('insights', {})
            growth = insights.get('total_growth_percent', 0)
            result += f"  • {pred['company_name']}: {growth:.1f}% growth\n"
    
    if failed:
        result += f"\n**Failed Predictions:** {len(failed)}\n"
        for pred in failed[:2]:
            result += f"  • {pred['company_name']}: {pred.get('error', 'Unknown error')}\n"
    
    result += f"\n**Available Analysis:**\n"
    result += f"  • Ask about specific companies\n"
    result += f"  • Growth pattern analysis\n"
    result += f"  • Model performance review\n"
    result += f"  • Confidence interval analysis\n"
    
    return result

# Create LangChain tools
prediction_analysis_tool = Tool(
    name="PredictionAnalyzer",
    func=analyze_recent_predictions,
    description="Analyze recent predictions. Can provide summaries, company-specific analysis, growth patterns, model performance, and confidence intervals. Use this when users ask about their predictions, forecast analysis, or model performance."
)

