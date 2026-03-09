import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from api.models.schemas import ChatRequest

load_dotenv()

try:
    with open("gpt_info.json") as f:
        GPT_INFO = json.load(f)
except Exception:
    GPT_INFO = {}

llm = ChatOpenAI(temperature=0.3, openai_api_key=os.getenv("OPENAI_API_KEY"))


def _build_prediction_context() -> str:
    """Load the latest prediction from memory and format it as LLM context."""
    try:
        from prediction_memory import prediction_memory
        recent = prediction_memory.get_recent_predictions(1)
        if not recent:
            return ""

        pred = recent[0]
        if not pred.get("success"):
            return ""

        input_data = pred.get("input_data", {})
        insights = pred.get("insights", {})
        forecasts = pred.get("predictions", [])
        trend = pred.get("trend_analysis") or {}
        method = pred.get("prediction_method", "Unknown")
        edge = pred.get("edge_case_analysis") or {}

        lines = [
            "=== LATEST PREDICTION (use this to answer the user's questions) ===",
            f"Company: {pred.get('company_name', 'Unknown')}",
            f"Prediction method: {method}",
            f"Model: {pred.get('model_used', 'Unknown')}",
            "",
            "Input (2023 quarterly ARR):",
            f"  Q1: ${input_data.get('q1_arr', 0):,.0f}",
            f"  Q2: ${input_data.get('q2_arr', 0):,.0f}",
            f"  Q3: ${input_data.get('q3_arr', 0):,.0f}",
            f"  Q4: ${input_data.get('q4_arr', 0):,.0f}",
            f"  Sector: {input_data.get('sector', 'N/A')}",
            f"  Headcount: {input_data.get('headcount', 'N/A')}",
        ]

        if forecasts:
            lines.append("")
            lines.append("Forecast (2024 predicted quarters):")
            for f in forecasts:
                lines.append(
                    f"  {f.get('quarter','?')}: ${f.get('predicted_arr',0):,.0f} "
                    f"(YoY {f.get('yoy_growth_percent',0):+.1f}%, "
                    f"range ${f.get('pessimistic_arr',0):,.0f}-${f.get('optimistic_arr',0):,.0f})"
                )

        current_arr = insights.get("current_arr", 0)
        final_arr = insights.get("predicted_final_arr", 0)
        growth = insights.get("total_growth_percent", 0)
        lines.append("")
        lines.append(f"Overall: ${current_arr:,.0f} -> ${final_arr:,.0f} ({growth:+.1f}% total growth)")
        lines.append(f"Data provided: {insights.get('tier_used', 'Tier 1 Only')}")
        lines.append(f"Confidence band: +/-10% on all quarters")

        if trend:
            lines.append("")
            lines.append("Trend detection:")
            lines.append(f"  Type: {trend.get('trend_type', 'N/A')}")
            lines.append(f"  Confidence: {trend.get('confidence', 'N/A')}")
            lines.append(f"  Reason: {trend.get('reason', 'N/A')}")
            metrics = trend.get("metrics", {})
            if metrics:
                lines.append(f"  Simple growth Q1->Q4: {metrics.get('simple_growth', 0):.2%}")
                lines.append(f"  Recent momentum Q3->Q4: {metrics.get('recent_momentum', 0):.2%}")
                lines.append(f"  Volatility: {metrics.get('volatility', 0):.3f}")

        if edge:
            lines.append("")
            lines.append("Edge-case health assessment:")
            lines.append(f"  Health tier: {edge.get('health_tier', 'N/A')}")
            lines.append(f"  Score: {edge.get('health_score', 'N/A')}")
            lines.append(f"  Reasoning: {edge.get('reasoning', 'N/A')}")
            lines.append(f"  Key assumption: {edge.get('key_assumption', 'N/A')}")

        lines.append("=== END OF PREDICTION ===")
        return "\n".join(lines)

    except Exception:
        return ""


def handle_chat(request: ChatRequest):
    """Context-aware chat that uses the LLM with full prediction data."""
    message = request.message.lower()
    name = request.name
    history = request.history or []

    # --- Static branches (no LLM needed) ---

    algorithm_keywords = [
        'how does', 'how it works', 'algorithm', 'model works',
        'feature injection', 'intelligent feature', 'tier based',
        'lightgbm', 'gradient boosted',
    ]
    if any(kw in message for kw in algorithm_keywords):
        return _algorithm_response()

    feature_keywords = [
        'magic number', 'burn multiple', 'rule of 40', 'gross margin',
        'headcount', 'customers', 'churn', 'expansion',
    ]
    if any(kw in message for kw in feature_keywords):
        return _feature_response(message)

    if 'csv' in message or 'upload' in message or 'file' in message:
        return {
            "response": (
                "You can analyze CSV data using the `/predict_csv` endpoint, which provides "
                "tier-based forecasting with confidence intervals and intelligent feature completion.\n\n"
                "Or just tell me your quarterly ARR numbers directly in chat!"
            )
        }

    # --- LLM path: everything else, with full prediction context ---

    prediction_context = _build_prediction_context()

    from api.services.macro_analysis import get_macro_analysis
    macro_data = get_macro_analysis()
    gprh = macro_data.get('gprh', {}).get('traffic_light', 'Unknown')
    vix = macro_data.get('vix', {}).get('traffic_light', 'Unknown')
    move = macro_data.get('move', {}).get('traffic_light', 'Unknown')
    bvp = macro_data.get('bvp', {}).get('traffic_light', 'Unknown')

    greeting = f"Hi {name}!" if name else "Hi!"
    project_info = GPT_INFO.get("project_info", {})
    project_info_str = " ".join([
        f"Creator: {project_info.get('creator', '')}.",
        f"Project name: {project_info.get('project_name', '')}.",
        f"Purpose: {project_info.get('purpose', '')}.",
        f"Contact: {project_info.get('contact', '')}."
    ])

    system_prompt = f"""{greeting} You are a VC-focused AI financial forecasting assistant.

You help investors and founders understand ARR forecasts produced by a hybrid ML system
(LightGBM + rule-based health assessment for edge cases, R² ≈ 0.80, ±10% confidence bands).

Current macro regime: GPRH={gprh}, VIX={vix}, MOVE={move}, BVP SaaS index={bvp}.
Project info: {project_info_str}

{prediction_context}

INSTRUCTIONS:
- If prediction data is shown above, use it to answer the user's question with specific numbers.
- When the user challenges a forecast ("shouldn't growth be higher?"), reason about it using
  the trend detection data, the input trajectory, and the prediction method.
- Be concise, professional, and specific. Refer to actual numbers, not generalities.
- If no prediction data is available, offer to help the user make one via the /tier_based_forecast
  endpoint or by providing their quarterly ARR data.
- You can explain SaaS metrics (Magic Number, Burn Multiple, Rule of 40, etc.) if asked.
- For algorithm questions, explain the 3-stage system: tier-based input → intelligent feature
  completion (top-50 peer matching) → LightGBM multi-horizon prediction.
"""

    conversation = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if "role" in msg and "content" in msg:
            conversation.append({"role": msg["role"], "content": msg["content"]})
    conversation.append({"role": "user", "content": request.message})

    response = llm.invoke(conversation).content
    return {"response": response}


def _algorithm_response() -> dict:
    """Static response explaining the algorithm from gpt_info.json."""
    algorithm_info = GPT_INFO.get("algorithm_explanation", {})
    model_info = GPT_INFO.get("model_details", {})

    stage1 = algorithm_info.get('stage_1_tier_based_input', {})
    stage2 = algorithm_info.get('stage_2_intelligent_feature_completion', {})
    stage3 = algorithm_info.get('stage_3_modeling', {})
    confidence = algorithm_info.get('confidence_intervals', {})

    r = f"🤖 **How Our Enhanced Tier-Based Prediction System Works**\n\n"
    r += f"**Overview:** {algorithm_info.get('overview', 'Our system uses advanced machine learning to predict ARR growth.')}\n\n"
    r += f"**🎯 Stage 1: Tier-Based Input**\n"
    r += f"• {stage1.get('description', 'Minimal required data with intelligent defaults')}\n"
    r += f"• {stage1.get('benefit', 'Reduces user burden while maintaining accuracy')}\n\n"
    r += f"**🧠 Stage 2: Intelligent Feature Completion**\n"
    r += f"• {stage2.get('description', 'Advanced pattern matching to infer missing features')}\n"
    r += f"• **Process:** {' → '.join(stage2.get('process', ['Company profiling', 'Similarity matching', 'Feature inference']))}\n"
    r += f"• **Features Created:** {len(stage2.get('features_created', []))}+ engineered features\n\n"
    r += f"**⚡ Stage 3: LightGBM Modeling**\n"
    r += f"• **Algorithm:** {stage3.get('algorithm', 'LightGBM')}\n"
    r += f"• **Why LightGBM:** {stage3.get('why_lightgbm', 'Excellent for tabular data and complex relationships')}\n"
    r += f"• **Training Data:** {stage3.get('training_data', '500+ VC-backed companies')}\n"
    r += f"• **Features:** {stage3.get('feature_count', '152')} engineered features per prediction\n\n"
    r += f"**📊 Confidence Intervals**\n"
    r += f"• **Method:** {confidence.get('method', '±10% uncertainty bands')}\n"
    r += f"• **Rationale:** {confidence.get('rationale', 'Based on model performance and business uncertainty')}\n\n"

    features = stage2.get('features_created', [])
    if features:
        r += f"**🎯 Key Features:**\n"
        for feat in features[:4]:
            r += f"• {feat}\n"

    perf = model_info.get('performance', {})
    r += f"\n**📈 Performance:** R² = {perf.get('r2', 0.7966):.1%} with {perf.get('confidence_intervals', '±10%')} confidence intervals"
    return {"response": r}


def _feature_response(message: str) -> dict:
    """Static response for SaaS metric definitions."""
    explanations = {
        'magic number': "The Magic Number measures sales efficiency: Net New ARR ÷ Sales & Marketing spend. Above 1.0 is excellent, 0.5-1.0 is good, below 0.5 needs improvement.",
        'burn multiple': "Burn Multiple shows cash efficiency: Net Burn ÷ Net New ARR. Below 1.0 is great, 1.0-2.0 is acceptable, above 2.0 is concerning.",
        'rule of 40': "Rule of 40 combines growth + profitability: Growth Rate + Profit Margin. Should be ≥40% for healthy SaaS companies.",
        'gross margin': "Gross Margin = (Revenue - COGS) ÷ Revenue. SaaS companies typically aim for 70-90% gross margins.",
        'headcount': "Employee count affects operational efficiency and burn rate. ARR per headcount is a key efficiency metric.",
        'customers': "Customer count and expansion/churn rates are crucial for growth sustainability.",
        'churn': "Customer churn rate should typically be <5% annually for healthy SaaS companies.",
        'expansion': "Expansion revenue from existing customers is often more efficient than new customer acquisition.",
    }
    for keyword, explanation in explanations.items():
        if keyword in message:
            return {"response": f"Great question about {keyword}! {explanation} Would you like me to help you calculate this metric for your company?"}
    return {"response": "I can help explain any SaaS metrics! Which specific metric would you like to know more about? Common ones include Magic Number, Burn Multiple, Rule of 40, Gross Margin, and more."}
