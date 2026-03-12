import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

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
        lines.append(f"Confidence band: {insights.get('confidence_intervals', '±10% on all predictions')}")

        if trend:
            lines.append("")
            lines.append("Trend detection:")
            lines.append(f"  Type: {trend.get('trend_type', 'N/A')}")
            lines.append(f"  Confidence: {trend.get('confidence', 'N/A')}")
            lines.append(f"  Reason: {trend.get('reason', 'N/A')}")
            metrics = trend.get("metrics", {})
            if metrics:
                lines.append(f"  Simple growth Q1->Q4: {metrics.get('simple_growth', 0):.2%}")
                qoq = metrics.get("qoq_growth", [])
                if len(qoq) == 3:
                    lines.append(f"  QoQ growth rates: Q1->Q2 {qoq[0]:+.2%}, Q2->Q3 {qoq[1]:+.2%}, Q3->Q4 {qoq[2]:+.2%}")
                lines.append(f"  Recent momentum (Q3->Q4): {metrics.get('recent_momentum', 0):.2%}")
                lines.append(f"  Volatility (std dev of QoQ rates): {metrics.get('volatility', 0):.3f}")
                lines.append(f"  Acceleration pattern: {metrics.get('acceleration', 'N/A')}")

        if edge:
            lines.append("")
            lines.append("Health scorecard:")
            lines.append(f"  Health tier: {edge.get('health_tier', 'N/A')}")
            lines.append(f"  Score: {edge.get('health_score', 'N/A')}/100")

            estimated = edge.get("estimated_metrics") or []
            if estimated:
                lines.append(f"  ESTIMATED metrics (not provided by user, inferred from ARR pattern): {', '.join(estimated)}")

            hm = edge.get("health_metrics") or {}
            if hm:
                def _tag(key):
                    return " [ESTIMATED]" if key in estimated else " [PROVIDED]"

                lines.append("")
                lines.append("Scoring metrics (5 pillars, 100 points total):")
                lines.append(f"  1. ARR Growth YoY: {hm.get('arr_growth_yoy_pct', 'N/A'):.1f}%  [25 pts — >=40% top quartile, >=15% moderate]{_tag('arr_growth')}")
                lines.append(f"  2. Net Revenue Retention (NRR): {hm.get('nrr', 'N/A'):.1f}%  [25 pts — >=120% excellent, >=100% good]{_tag('nrr')}")
                lines.append(f"  3. CAC Payback: {hm.get('cac_payback_months', 'N/A'):.0f} months  [20 pts — <=18mo top quartile, <=36mo good]{_tag('cac_payback')}")
                lines.append(f"  4. Rule of 40: {hm.get('rule_of_40', 'N/A'):.1f}% (growth {hm.get('arr_growth_yoy_pct', 0):.1f}% + EBITDA margin {hm.get('ebitda_margin', 0):.1f}%)  [20 pts — >=40% target]{_tag('rule_of_40')}")
                lines.append(f"  5. Runway: {hm.get('runway_months', 'N/A'):.0f} months  [10 pts — >=18mo strong, >=12mo adequate]{_tag('runway')}")
                lines.append(f"  Gross Margin: {hm.get('gross_margin', 'N/A')}%")

            sb = edge.get("scoring_breakdown") or {}
            strengths = sb.get("strengths", [])
            weaknesses = sb.get("weaknesses", [])
            benchmarks_met = sb.get("benchmarks_met", [])
            benchmarks_missed = sb.get("benchmarks_missed", [])

            if strengths:
                lines.append("")
                lines.append("Strengths:")
                for s in strengths:
                    lines.append(f"  + {s}")
            if weaknesses:
                lines.append("Weaknesses:")
                for w in weaknesses:
                    lines.append(f"  - {w}")
            if benchmarks_met:
                lines.append("Benchmarks met:")
                for b in benchmarks_met:
                    lines.append(f"  ✓ {b}")
            if benchmarks_missed:
                lines.append("Benchmarks missed:")
                for b in benchmarks_missed:
                    lines.append(f"  ✗ {b}")

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

    try:
        from api.services.macro_analysis import get_macro_analysis
        macro_data = get_macro_analysis()
    except Exception:
        macro_data = {}
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

=== SYSTEM METHODOLOGY (cite these specifics when the user asks "how" or "why") ===

TREND DETECTION (6-factor analysis on the 4 input quarters):
  Inputs computed from quarterly ARR (Q1-Q4):
    - simple_growth = (Q4 - Q1) / Q1
    - QoQ growth rates: qoq1=(Q2-Q1)/Q1, qoq2=(Q3-Q2)/Q2, qoq3=(Q4-Q3)/Q3
    - volatility = standard deviation of [qoq1, qoq2, qoq3]  (numpy.std)
    - recent_momentum = qoq3  (the most recent quarter-over-quarter change)
    - acceleration = "accelerating" if qoq3>qoq2>qoq1, "decelerating" if qoq3<qoq2<qoq1, else "irregular"
    - consistency checks: all_positive (all QoQ>0), all_negative (all QoQ<0)
  Classification rules (checked in priority order):
    1. CONSISTENT_DECLINE: all QoQ negative, OR (simple_growth < -15% AND momentum < -10%)
    2. TREND_REVERSAL: overall positive but recent momentum < -15%, or vice versa
    3. VOLATILE_IRREGULAR: volatility > 0.20 (i.e. std dev of QoQ rates exceeds 20pp)
    4. FLAT_STAGNANT: |simple_growth| < 10% AND |avg QoQ| < 5%
    5. CONSISTENT_GROWTH: all QoQ positive AND simple_growth > 20%
    6. MODERATE_GROWTH: simple_growth > 0 (catch-all positive)
    7. UNCLEAR: default
  Routing: types 1-4 and 7 → Rule-Based Health Assessment; types 5-6 → ML model (LightGBM).

HEALTH SCORING (5 pillars, 100 points total, benchmarks from McKinsey/BCG/BVP):
  1. ARR Growth YoY (25 pts): >=40% = 25pts (top quartile), >=15% = 15pts, else 5pts
  2. NRR (25 pts): >=120% = 25pts (excellent), >=100% = 15pts, else 5pts
     If user didn't provide churn/expansion, NRR is estimated: 105% if growing, 95% if declining.
  3. CAC Payback (20 pts): <=18mo = 20pts (top quartile), <=36mo = 12pts, else 5pts
     Estimated from growth rate if S&M/customers not provided.
  4. Rule of 40 (20 pts): growth% + EBITDA margin%. >=40 = 20pts, >=30 = 12pts, else 5pts
     EBITDA margin estimated as gross_margin - 35% if gross margin provided, else default -10%.
  5. Runway (10 pts): >=18mo = 10pts, >=12mo = 6pts, else 2pts
  Tiers: >=75 pts = HIGH, >=50 pts = MODERATE, else LOW.

GROWTH PROJECTION (applied after health tier classification):
  HIGH tier: projected annual growth = 30-40% (with deceleration as company scales)
  MODERATE tier: projected annual growth = 10-15% (aligned with industry median ~22%)
  LOW tier: projected annual growth = -5% to +5% (conservative, factoring in weak fundamentals)
  Quarterly: annual rate / 4, with 2% deceleration per successive quarter.

=== END METHODOLOGY ===

INSTRUCTIONS:
- ONLY use data that is explicitly shown in the prediction context above. If a metric, score,
  or value is NOT listed above, say "this metric was not computed for this prediction" — do NOT
  invent, estimate, or infer values that are not shown. This is critical for user trust.
- Metrics tagged [ESTIMATED] were not provided by the user and were inferred from the ARR
  pattern. Always disclose this when citing them: e.g. "NRR is estimated at 107% (inferred
  from the growth trajectory, not provided by the user)."
- If prediction data is shown above, use it to answer the user's question with specific numbers.
- When the user asks HOW something is calculated, cite the exact formula from the methodology
  above and show the calculation with the actual values from this prediction. For example, if
  asked about volatility, show: "Volatility = std([qoq1, qoq2, qoq3]) = std([X%, Y%, Z%]) = 0.227".
- When the user challenges a forecast ("shouldn't growth be higher?"), reason about it using
  the trend detection data, the input trajectory, and the prediction method.
- When the user asks "why" or wants deeper analysis, reference the 5 scoring pillars
  (ARR Growth, NRR, CAC Payback, Rule of 40, Runway), their actual values, and the
  industry benchmarks they were scored against. Explain which pillars earned high/low points
  and how that drove the health tier and growth projection.
- Mention strengths and weaknesses by name. If metrics were estimated (not provided by the
  user), note that they were inferred from the ARR pattern, not from actual data.
- Be concise, professional, and specific. Refer to actual numbers, not generalities.
  NEVER give vague answers like "it could be due to various factors". Always ground your
  response in the specific data and formulas available.
- If no prediction data is available, offer to help the user make one via the /tier_based_forecast
  endpoint or by providing their quarterly ARR data.
"""

    conversation = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if "role" in msg and "content" in msg:
            conversation.append({"role": msg["role"], "content": msg["content"]})
    conversation.append({"role": "user", "content": request.message})

    try:
        response = llm.invoke(conversation).content
    except Exception as e:
        logger.error(f"LLM invoke failed: {e}", exc_info=True)
        raise
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
