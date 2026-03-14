# 🚀 Venture Prophet API Specification

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. **POST /tier_based_forecast** - Main Prediction Endpoint

**Request:**
```json
{
  "company_name": "RocketAI Technologies",
  "q1_arr": 2000000,
  "q2_arr": 2500000,
  "q3_arr": 3200000,
  "q4_arr": 4200000,
  "headcount": 25,
  "sector": "Data & Analytics",
  "tier2_metrics": {
    "gross_margin": 85,
    "sales_marketing": 1200000,
    "cash_burn": 800000,
    "customers": 150,
    "churn_rate": 2.0,
    "expansion_rate": 25.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "company_name": "RocketAI Technologies",
  "model_used": "Enhanced Tier-Based Model with Confidence Intervals",
  "insights": {
    "current_arr": 4200000,
    "predicted_final_arr": 9884832,
    "total_growth_percent": 135.4,
    "final_yoy_growth_percent": 135.4,
    "tier_used": "Tier 1 + Tier 2",
    "model_accuracy": "R² = 0.7966 (79.66%)"
  },
  "forecast": [
    {
      "quarter": "Q1 2024",
      "predicted_arr": 5004492,
      "pessimistic_arr": 4504043,
      "optimistic_arr": 5504941,
      "qoq_growth_percent": 19.2,
      "confidence_interval": "±10%"
    },
    {
      "quarter": "Q2 2024",
      "predicted_arr": 5842455,
      "pessimistic_arr": 5258209,
      "optimistic_arr": 6426700,
      "qoq_growth_percent": 16.7,
      "confidence_interval": "±10%"
    },
    {
      "quarter": "Q3 2024",
      "predicted_arr": 7396353,
      "pessimistic_arr": 6656717,
      "optimistic_arr": 8135988,
      "qoq_growth_percent": 26.6,
      "confidence_interval": "±10%"
    },
    {
      "quarter": "Q4 2024",
      "predicted_arr": 9884832,
      "pessimistic_arr": 8896349,
      "optimistic_arr": 10873315,
      "qoq_growth_percent": 33.6,
      "confidence_interval": "±10%"
    }
  ]
}
```

### 2. **POST /chat** - Conversational AI

**Request:**
```json
{
  "message": "How does the algorithm work?",
  "name": "Test User",
  "preferred_model": "lightgbm",
  "history": []
}
```

**Response:**
```json
{
  "response": "🤖 **How Our Enhanced Tier-Based Prediction System Works**\n\n**Overview:** Our Enhanced Tier-Based Prediction System uses a sophisticated 3-stage approach..."
}
```

### 3. **GET /health** - Health Check

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_status": "loaded",
  "api_status": "running"
}
```

### 4. **GET /model_info** - Model Information

**Response:**
```json
{
  "model_name": "Enhanced Tier-Based Financial Forecasting Model",
  "accuracy": "R² = 0.7966 (79.66%)",
  "features": 152,
  "training_data_size": "5085 records",
  "status": "Production Ready"
}
```

## Key Features

- **QoQ Growth Display**: Each quarter shows quarter-over-quarter growth
- **Confidence Intervals**: ±10% uncertainty bands
- **Tier-Based Input**: Tier 1 (Required) + Tier 2 (Optional)
- **Prediction Analysis**: Chat can analyze recent predictions
- **Algorithm Explanation**: Detailed system explanation available

## Valid Sectors
```
"Cyber Security"
"Data & Analytics" 
"Infrastructure & Network"
"Communication & Collaboration"
"Marketing & Customer Experience"
"Other"
```
