#!/usr/bin/env python3
"""
Simplified Prediction Service for Tier-Based API
===============================================

Only includes the new tier-based forecasting endpoint to avoid import issues.
"""

import sys
import os
import pandas as pd
import io
from prediction_memory import add_tier_based_prediction, add_csv_prediction
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.models.schemas import TierBasedRequest


def generate_prediction_narrative(result: dict) -> str:
    """Generate an LLM-powered narrative that explains the prediction in plain language."""
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        company = result.get("company_name", "the company")
        insights = result.get("insights", {})
        forecast = result.get("forecast", [])
        trend = result.get("trend_analysis", {})
        method = result.get("prediction_method", "ML")
        edge = result.get("edge_case_analysis")

        current_arr = insights.get("current_arr", 0)
        final_arr = insights.get("predicted_final_arr", 0)
        growth = insights.get("total_growth_percent", 0)
        tier_used = insights.get("tier_used", "Tier 1 Only")

        quarters_summary = "\n".join(
            f"  {f['quarter']}: ${f['predicted_arr']:,.0f} "
            f"(YoY {f['yoy_growth_percent']:+.1f}%, "
            f"range ${f['pessimistic_arr']:,.0f}-${f['optimistic_arr']:,.0f})"
            for f in forecast
        )

        trend_desc = ""
        if trend:
            trend_desc = (
                f"Trend detected: {trend.get('trend_type', 'N/A')} "
                f"(confidence: {trend.get('confidence', 'N/A')}). "
                f"Reason: {trend.get('reason', 'N/A')}."
            )

        edge_desc = ""
        if edge:
            edge_desc = (
                f"Health assessment: tier={edge.get('health_tier','N/A')}, "
                f"reasoning={edge.get('reasoning','N/A')}."
            )

        prompt = f"""You are a VC analyst assistant. A prediction has just been made for "{company}".
Provide a concise (3-5 paragraph) plain-language analysis. Be specific about the numbers.

DATA:
- Current Q4 ARR: ${current_arr:,.0f}
- Predicted Q4-ahead ARR: ${final_arr:,.0f}
- Total growth: {growth:+.1f}%
- Data provided: {tier_used}
- Prediction method: {method}
- {trend_desc}
- {edge_desc}

Quarterly forecast:
{quarters_summary}

Cover:
1. What the forecast says about this company's trajectory
2. Why the system chose the {method} path (based on the trend detection)
3. Key risks or opportunities visible in the numbers
4. How the ±10% confidence band should be interpreted for decision-making

Keep the tone professional but accessible. Do not repeat raw numbers excessively."""

        return llm.invoke(prompt).content
    except Exception as e:
        return f"Automated analysis unavailable: {e}"


def perform_tier_based_forecast(request: TierBasedRequest):
    """Tier-based forecasting using the hybrid ML+GPT system with trend detection."""
    try:
        # Import the hybrid prediction system
        from hybrid_prediction_system import HybridPredictionSystem
        
        # Prepare Tier 1 data
        tier1_data = {
            'q1_arr': request.q1_arr,
            'q2_arr': request.q2_arr,
            'q3_arr': request.q3_arr,
            'q4_arr': request.q4_arr,
            'headcount': request.headcount,
            'sector': request.sector
        }
        
        # Prepare Tier 2 data if provided
        tier2_data = None
        if request.tier2_metrics:
            tier2_data = {
                'gross_margin': request.tier2_metrics.gross_margin,
                'sales_marketing': request.tier2_metrics.sales_marketing,
                'cash_burn': request.tier2_metrics.cash_burn,
                'customers': request.tier2_metrics.customers,
                'churn_rate': request.tier2_metrics.churn_rate,
                'expansion_rate': request.tier2_metrics.expansion_rate
            }
        
        # Initialize hybrid system
        hybrid_system = HybridPredictionSystem()
        
        # Get predictions with trend analysis
        predictions, metadata = hybrid_system.predict_with_hybrid(tier1_data, tier2_data)
        
        # Determine confidence band from trend type
        trend_type = metadata.get('trend_analysis', {}).get('trend_type', '')
        band_pct = 25 if trend_type == 'VOLATILE_IRREGULAR' else 10

        # Format response with QoQ growth calculated from honest model predictions
        forecast_results = []
        current_arr = request.q4_arr  # Start from Q4 2023
        
        for pred in predictions:
            # Calculate QoQ growth from previous quarter (honest approach)
            qoq_growth = ((pred['ARR'] - current_arr) / current_arr) * 100 if current_arr > 0 else 0
            
            forecast_results.append({
                "quarter": pred['Quarter'],
                "predicted_arr": pred['ARR'],
                "pessimistic_arr": pred['Pessimistic_ARR'],
                "optimistic_arr": pred['Optimistic_ARR'],
                "qoq_growth_percent": qoq_growth,
                "yoy_growth_percent": pred['YoY_Growth_Percent'],
                "confidence_interval": f"±{band_pct}%"
            })
            
            current_arr = pred['ARR']  # Update for next quarter
        
        # Calculate insights
        current_arr = request.q4_arr
        final_predicted_arr = predictions[-1]['ARR']
        total_growth = ((final_predicted_arr - current_arr) / current_arr) * 100
        
        # Calculate cumulative ARR for 2023 (user input)
        cumulative_arr_2023 = request.q1_arr + request.q2_arr + request.q3_arr + request.q4_arr
        
        # Calculate cumulative ARR for 2024 (model predictions)
        cumulative_arr_2024 = sum([pred['ARR'] for pred in predictions])
        
        # Calculate total ARR (2023 + 2024 cumulative)
        total_arr = cumulative_arr_2023 + cumulative_arr_2024
        
        insights = {
            'company_name': request.company_name or "Anonymous Company",
            'current_arr': current_arr,
            'predicted_final_arr': final_predicted_arr,
            'total_growth_percent': total_growth,
            'final_yoy_growth_percent': predictions[-1]['YoY_Growth_Percent'],
            'tier_used': 'Tier 1 + Tier 2' if tier2_data else 'Tier 1 Only',
            'model_accuracy': f"R² = {hybrid_system.ml_system.model_data.get('overall_r2', 0.85):.4f}",
            'confidence_intervals': f'±{band_pct}% on all predictions',
            'user_input_arr_2023': {
                'q1': request.q1_arr,
                'q2': request.q2_arr,
                'q3': request.q3_arr,
                'q4': request.q4_arr,
                'cumulative': cumulative_arr_2023
            },
            'predicted_arr_2024': {
                'q1': predictions[0]['ARR'],
                'q2': predictions[1]['ARR'],
                'q3': predictions[2]['ARR'],
                'q4': predictions[3]['ARR'],
                'cumulative': cumulative_arr_2024
            },
            'total_arr': total_arr
        }
        
        # Extract trend analysis from metadata
        trend_analysis = metadata['trend_analysis']
        
        result = {
            "success": True,
            "company_name": request.company_name or "Anonymous Company",
            "insights": insights,
            "forecast": forecast_results,
            "model_used": f"Hybrid System ({metadata['prediction_method']})",
            "prediction_method": metadata['prediction_method'],
            "trend_analysis": {
                "trend_type": trend_analysis['trend_type'],
                "confidence": trend_analysis['confidence'],
                "reason": trend_analysis['reason'],
                "user_message": trend_analysis['user_message'],
                "metrics": {
                    "simple_growth": trend_analysis['metrics']['simple_growth'],
                    "qoq_growth": trend_analysis['metrics']['qoq_growth'],
                    "recent_momentum": trend_analysis['metrics']['recent_momentum'],
                    "volatility": trend_analysis['metrics']['volatility'],
                    "acceleration": trend_analysis['metrics']['acceleration']
                }
            },
            "tier_analysis": {
                "tier1_provided": True,
                "tier2_provided": tier2_data is not None,
                "tier2_metrics_count": len([v for v in (tier2_data or {}).values() if v is not None]) if tier2_data else 0
            }
        }
        
        # Always attach health scorecard so the chatbot has real numbers
        ha = metadata.get('health_assessment') or {}
        hm = metadata.get('health_metrics') or {}
        estimated_list = metadata.get('estimated_metrics', [])
        if not estimated_list:
            estimated_flags = hm.get('_estimated_flags', {})
            flag_labels = {
                'nrr': 'NRR (churn/expansion not provided)',
                'cac_payback': 'CAC Payback (S&M/customers not provided)',
                'rule_of_40': 'Rule of 40 (gross margin not provided)',
                'runway': 'Runway (not provided)',
                'cash_burn': 'Cash burn (not provided)',
            }
            estimated_list = [flag_labels.get(k, k) for k, v in estimated_flags.items() if v]

        if ha or hm:
            is_edge = metadata.get('prediction_method') == 'Rule-Based Health Assessment'
            result['edge_case_analysis'] = {
                "method": metadata.get('prediction_method', 'ML_Model'),
                "health_tier": metadata.get('health_tier', 'N/A'),
                "health_score": ha.get('score', 'N/A'),
                "reasoning": metadata.get('reasoning', 'Forecasted via ML model') if is_edge else 'Health scorecard computed alongside ML forecast',
                "confidence": metadata.get('confidence', 'N/A'),
                "key_assumption": metadata.get('key_assumption', 'N/A'),
                "health_metrics": {
                    "arr_growth_yoy_pct": hm.get('arr_growth_yoy_percent'),
                    "nrr": hm.get('nrr'),
                    "cac_payback_months": hm.get('cac_payback_months'),
                    "rule_of_40": hm.get('rule_of_40'),
                    "gross_margin": hm.get('gross_margin'),
                    "ebitda_margin": hm.get('ebitda_margin'),
                    "runway_months": hm.get('runway_months'),
                    "cash_burn": hm.get('cash_burn'),
                    "recent_momentum_qoq": hm.get('recent_momentum'),
                },
                "scoring_breakdown": {
                    "strengths": ha.get('strengths', []),
                    "weaknesses": ha.get('weaknesses', []),
                    "benchmarks_met": ha.get('benchmarks_met', []),
                    "benchmarks_missed": ha.get('benchmarks_missed', []),
                },
                "estimated_metrics": estimated_list,
            }
        
        # Generate LLM narrative alongside the structured prediction
        result["analysis_narrative"] = generate_prediction_narrative(result)
        
        # Store prediction in memory for chat analysis
        input_data = {
            "q1_arr": request.q1_arr, "q2_arr": request.q2_arr, 
            "q3_arr": request.q3_arr, "q4_arr": request.q4_arr,
            "headcount": request.headcount, "sector": request.sector,
            "tier2_metrics": tier2_data
        }
        add_tier_based_prediction(result, input_data)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Tier-based forecast failed: {str(e)}",
            "company_name": request.company_name or "Anonymous Company"
        }

def predict_from_csv(file_content: bytes, company_name: str = None):
    """Predict from CSV upload using the tier-based system."""
    try:
        # Import pandas here to ensure it's available
        import pandas as pd
        import io
        
        # Read CSV content
        csv_content = file_content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Validate CSV structure
        required_columns = ['Quarter', 'ARR_End_of_Quarter', 'Headcount', 'Gross_Margin_Percent']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                "success": False,
                "error": f"Missing required columns: {missing_columns}. Required: {required_columns}",
                "company_name": company_name or "CSV Upload Company"
            }
        
        # Extract Tier 1 data from CSV
        # Assume CSV has 4 quarters of data (Q1-Q4 2023)
        if len(df) < 4:
            return {
                "success": False,
                "error": "CSV must contain at least 4 quarters of data (Q1-Q4 2023)",
                "company_name": company_name or "CSV Upload Company"
            }
        
        # Get the last 4 quarters (assuming they're in chronological order)
        last_4_quarters = df.tail(4)
        
        headcount = int(last_4_quarters.iloc[-1]['Headcount'])
        
        from api.models.schemas import VALID_SECTORS
        inferred_sector = 'Other'
        if 'Sector' in df.columns:
            csv_sector = str(last_4_quarters.iloc[-1]['Sector']).strip()
            if csv_sector in VALID_SECTORS:
                inferred_sector = csv_sector
        
        tier1_data = {
            'q1_arr': last_4_quarters.iloc[0]['ARR_End_of_Quarter'],
            'q2_arr': last_4_quarters.iloc[1]['ARR_End_of_Quarter'],
            'q3_arr': last_4_quarters.iloc[2]['ARR_End_of_Quarter'],
            'q4_arr': last_4_quarters.iloc[3]['ARR_End_of_Quarter'],
            'headcount': headcount,
            'sector': inferred_sector  # Intelligently inferred sector
        }
        
        # Extract Tier 2 data if available
        tier2_data = None
        latest_row = last_4_quarters.iloc[-1]
        
        # Check if we have additional metrics in the CSV
        if 'Net_Profit_Loss_Margin_Percent' in df.columns:
            tier2_data = {
                'gross_margin': latest_row.get('Gross_Margin_Percent', 75),
                'sales_marketing': latest_row.get('Sales_Marketing', tier1_data['q4_arr'] * 0.4),
                'cash_burn': latest_row.get('Cash_Burn', -tier1_data['q4_arr'] * 0.3),
                'customers': latest_row.get('Customers', int(tier1_data['q4_arr'] / 5000)),
                'churn_rate': latest_row.get('Churn_Rate', 5),
                'expansion_rate': latest_row.get('Expansion_Rate', 10)
            }
        
        # Create TierBasedRequest object
        from api.models.schemas import TierBasedRequest, Tier2Metrics
        
        tier2_metrics = None
        if tier2_data:
            tier2_metrics = Tier2Metrics(
                gross_margin=tier2_data.get('gross_margin'),
                sales_marketing=tier2_data.get('sales_marketing'),
                cash_burn=tier2_data.get('cash_burn'),
                customers=tier2_data.get('customers'),
                churn_rate=tier2_data.get('churn_rate'),
                expansion_rate=tier2_data.get('expansion_rate')
            )
        
        request = TierBasedRequest(
            company_name=company_name or "CSV Upload Company",
            q1_arr=tier1_data['q1_arr'],
            q2_arr=tier1_data['q2_arr'],
            q3_arr=tier1_data['q3_arr'],
            q4_arr=tier1_data['q4_arr'],
            headcount=tier1_data['headcount'],
            sector=tier1_data['sector'],
            tier2_metrics=tier2_metrics if tier2_metrics else None
        )
        
        # Use the existing tier-based forecast function
        result = perform_tier_based_forecast(request)
        
        # Add CSV-specific information
        if result["success"]:
            result["csv_info"] = {
                "rows_processed": len(df),
                "quarters_used": 4,
                "tier2_metrics_extracted": tier2_data is not None,
                "csv_columns": list(df.columns)
            }
            
            # Store CSV prediction in memory for chat analysis
            input_data = {
                "csv_columns": list(df.columns),
                "rows_processed": len(df),
                "tier1_data": tier1_data,
                "tier2_data": tier2_data
            }
            add_csv_prediction(result, input_data)
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"CSV processing failed: {str(e)}",
            "company_name": company_name or "CSV Upload Company"
        }
