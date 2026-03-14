# Active vs Unused Files Analysis

## ✅ **ACTIVELY USED IN PRODUCTION** (via fastapi_app.py)

### Core API Files
- `fastapi_app.py` - Main API entry point (used in Render deployment)
- `api/routers/tier_predictions.py` - Tier-based prediction endpoints
- `api/routers/tier_system.py` - System info endpoints  
- `api/routers/macro.py` - Macro analysis endpoints
- `api/services/tier_prediction.py` - Tier prediction logic
- `api/services/tier_system.py` - System info logic
- `api/services/macro_analysis.py` - Macro analysis logic
- `api/services/prediction.py` - Chat handler
- `api/models/schemas.py` - Pydantic data models

### Prediction System
- `tier_based_prediction_system.py` - **Main prediction orchestrator**
- `intelligent_feature_completion_system.py` - **Feature completion and ML model**
- `lightgbm_financial_model.pkl` - **The actual ML model (11MB)**

### Supporting Files
- `prediction_memory.py` - Stores predictions for chat analysis
- `prediction_analysis_tools.py` - Tools for chat to analyze predictions
- `gpr_analysis.py` - Geopolitical risk analysis
- `vix_analysis.py` - Market volatility analysis  
- `move_analysis.py` - Bond market analysis
- `bvp_analysis.py` - BVP Cloud Index analysis

### Data & Config
- `202402_Copy.csv` - Training data
- `requirements.txt` - Dependencies
- `render.yaml` - Deployment config

---

## ❌ **NOT ACTIVELY USED** (Development/Archive Files)

### Standalone Systems (Not Connected to API)
- `fastapi_app_simple.py` ❌ - Alternative API without macro endpoints (unused, docs were outdated)
- `enhanced_guided_input.py` ❌ - Old input system (replaced by tier_based_prediction_system)
- `enhanced_simple_model.py` ❌ - Experimental model
- `financial_forecasting_model.py` ❌ - Old forecasting system
- `production_ready_system.py` ❌ - Old production system (replaced)
- `cumulative_arr_model.py` ❌ - Experimental ARR model
- `cumulative_arr_system.py` ❌ - Old cumulative system
- `improved_cumulative_arr_system.py` ❌ - Another experiment
- `direct_arr_prediction_system.py` ❌ - Alternative approach

### Analysis Scripts (Development Only)
- `analyze_cumulative_arr.py` ❌
- `analyze_model_accuracy.py` ❌
- `explain_model_metrics.py` ❌
- `debug_csv.py` ❌
- `deploy_to_lovable.py` ❌

### Test Files (Not Production)
- `test_*.py` - All test files
- `tests/` directory - All test scripts
- `sample_companies.csv` - Test data
- `test_company_*.csv` - Test data

### Unused Models (7 files)
- `corrected_arr_prediction_model.pkl` ❌
- `cumulative_arr_model.pkl` ❌
- `enhanced_financial_forecasting_model.pkl` ❌
- `fixed_financial_forecasting_model.pkl` ❌
- `lightgbm_cleaned_model.pkl` ❌
- `lightgbm_single_quarter_models.pkl` ❌
- `realistic_growth_forecasting_model.pkl` ❌

### Archive Directory
- `archive/` - Everything here is archived/unused

---

## 📊 **PRODUCTION DATA FLOW**

```
API Request
    ↓
fastapi_app.py
    ↓
api/routers/tier_predictions.py
    ↓
api/services/tier_prediction.py
    ↓
tier_based_prediction_system.py
    ↓
intelligent_feature_completion_system.py
    ↓
lightgbm_financial_model.pkl (THE MODEL)
    ↓
Response with predictions
```

---

## 🎯 **SUMMARY**

**Actually Used**: ~20 files
**Unused/Archive**: ~80+ files

The production system is actually quite clean - it only uses:
1. The tier-based API infrastructure
2. The intelligent feature completion system
3. One LightGBM model (11MB)
4. Macro analysis modules
5. Chat/memory support

Everything else (enhanced_guided_input.py, enhanced_simple_model.py, etc.) was from development/experimentation and is **not connected to the production API**.

