# VenturePredictor — Intelligent SaaS Forecasting for VC

A hybrid ML + rule-based forecasting system that predicts ARR (Annual Recurring Revenue) for SaaS companies across four future quarters. Built for venture capital workflows where data is sparse, trajectories vary wildly, and transparency matters.

## 🏗 Architecture Overview

The system routes each prediction through an intelligent pipeline:

1. **Tier-based input parsing** — separates required inputs (Tier 1) from optional advanced metrics (Tier 2)
2. **Intelligent Feature Completion** — imputes missing Tier 2 fields using peer-similarity matching and weighted medians
3. **Trend Detection** — classifies the company's ARR trajectory (growth, decline, volatile, reversal, flat) using a 6-factor analysis
4. **Hybrid Routing** — standard growth patterns go to the LightGBM ML model; edge cases (decline, volatility, stagnation) are routed to a Rule-Based Health Assessment
5. **Health Scorecard** — all predictions include a 100-point health score across five pillars (ARR Growth, NRR, CAC Payback, Rule of 40, Runway)
6. **Uncertainty Bands** — ±10% for standard trajectories, ±25% for volatile cases
7. **LLM Narrative** — GPT-powered natural language explanation of the forecast

## 📥 Input Contract

### Tier 1 (Required)
| Field | Type | Description |
|-------|------|-------------|
| `q1_arr` – `q4_arr` | float | Four consecutive quarters of ARR |
| `headcount` | int | Current employee count |
| `sector` | string | One of: Cyber Security, Data & Analytics, Infrastructure & Network, Communication & Collaboration, Marketing & Customer Experience, Other |

### Tier 2 (Optional)
| Field | Type | Description |
|-------|------|-------------|
| `gross_margin` | float | Gross margin (%) |
| `sales_marketing` | float | S&M spend ($) |
| `cash_burn` | float | Monthly cash burn ($) |
| `customers` | float | End-of-period customer count |
| `churn_rate` | float | Annual logo/revenue churn (%) |
| `expansion_rate` | float | Net expansion rate (%) |

When Tier 2 fields are missing, the Intelligent Feature Completion system fills them by finding the 50 most similar companies (by ARR scale, growth rate, and headcount) and computing weighted medians from that peer set.

## 🔀 Hybrid Forecasting

Not all companies should be predicted the same way. The **Trend Detector** evaluates six factors — simple growth, QoQ rates, momentum, consistency, volatility, and acceleration — against priority-ordered thresholds:

| Pattern | Route | Rationale |
|---------|-------|-----------|
| Consistent growth / moderate growth | ML Model (LightGBM) | Standard regime the model was trained on |
| Decline, reversal, flat, volatile, unclear | Rule-Based Health Assessment | Edge cases where growth-trained ML produces unreliable forecasts |

The Rule-Based path scores the company on five health pillars (benchmarked against McKinsey, BCG, BVP research), assigns a health tier (High / Moderate / Low), and applies tier-specific growth projection rules.

## 🤖 ML Model

- **Algorithm:** LightGBM (gradient-boosted decision trees)
- **Objective:** `regression_l1` (MAE loss, robust to outliers)
- **Output:** Four-quarter-ahead ARR predictions (multi-output regression)
- **Training data:** ~5,000 company-quarter observations across ~350 companies
- **Features:** ~150 engineered features (financial scalars, SaaS ratios, temporal lags, categorical encodings)
- **Validation:** Company-based holdout split (80/20) — no company appears in both train and test
- **Performance:** R² ≈ 0.80 (aggregated across horizons)

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/tier_based_forecast` | Primary forecasting endpoint (Tier 1 + optional Tier 2) |
| `POST` | `/predict_csv` | Batch predictions from CSV upload |
| `POST` | `/chat` | Conversational AI interface (LangChain + GPT) |
| `GET` | `/makro-analysis` | Macroeconomic context (GPRH, VIX, MOVE, BVP indices) |
| `GET` | `/model-info` | Model metadata and feature list |
| `GET` | `/health` | Service health check |

## 🌐 Macroeconomic Context Module

Forecasts are contextualised (not driven) by four macro indicators:

- **GPRH** — Geopolitical Risk Index
- **VIX** — Market volatility
- **MOVE** — Bond market volatility
- **BVP Cloud Index** — SaaS sector performance

These are presented as a decision-support overlay, not fed into the ML model.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable for chat functionality
export OPENAI_API_KEY=your_key_here

# Start the server
uvicorn fastapi_app:app --reload
```

### Example Request
```bash
curl -X POST "http://localhost:8000/tier_based_forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "Acme SaaS",
    "q1_arr": 1000000,
    "q2_arr": 1400000,
    "q3_arr": 2000000,
    "q4_arr": 2800000,
    "headcount": 70,
    "sector": "Data & Analytics",
    "tier2_metrics": {
      "gross_margin": 82.0,
      "churn_rate": 3.0,
      "expansion_rate": 25.0
    }
  }'
```

## 🛠 Tech Stack

- **API:** FastAPI + Uvicorn
- **ML:** LightGBM, scikit-learn
- **Chat:** LangChain + OpenAI GPT
- **Validation:** Pydantic
- **Deployment:** Render (`render.yaml`)

## 📁 Project Structure

```
├── fastapi_app.py                         # App entrypoint
├── api/
│   ├── routers/                           # FastAPI route handlers
│   ├── models/schemas.py                  # Pydantic request/response models
│   └── services/                          # Business logic (prediction, chat, macro)
├── hybrid_prediction_system.py            # Orchestrates ML vs. rule-based routing
├── trend_detector.py                      # 6-factor trend classification
├── rule_based_health_predictor.py         # Health scoring + deterministic forecasts
├── intelligent_feature_completion_system.py  # Peer-similarity imputation
├── tier_based_prediction_system.py        # Tier 1/2 input handling
├── financial_prediction.py                # LightGBM model loading + inference
├── prediction_analysis_tools.py           # LangChain tools for chat
├── *_analysis.py                          # Macro indicator fetchers (VIX, MOVE, BVP, GPRH)
└── archive/                               # Historical training scripts and analysis
```

## 📄 License

Academic project — Reichman University.
