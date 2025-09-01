# 🚀 Enhanced Financial Forecasting API

A production-ready financial forecasting API that predicts ARR (Annual Recurring Revenue) growth using machine learning models trained on real SaaS company data.

## ✨ **Key Features**

### **🎯 Smart Forecasting**
- **LightGBM Model**: High-performance gradient boosting for accurate predictions (single model approach)
- **Uncertainty Quantification**: ±10% confidence bands for realistic forecasts
- **Multi-Quarter Predictions**: Forecast 4 quarters ahead with detailed breakdowns

### **🔧 Flexible Input Modes**
- **Basic Mode**: Minimal inputs (company name, current ARR, net new ARR)
- **Enhanced Mode**: Optional sector/country/currency selection for better accuracy
- **Historical ARR**: Provide 4 quarters of historical data for improved predictions
- **Advanced Mode**: Override 14 key financial metrics with your own values

### **📊 Data-Driven Intelligence**
- **Adaptive Defaults**: Uses training data relationships for realistic estimates
- **Sector-Specific Patterns**: 7 main sectors covering 81% of companies
- **Geographic Context**: 4 main countries covering 87% of companies
- **Smart Imputation**: Intelligent feature filling based on company characteristics

### **🤖 Conversational AI**
- **Chat Interface**: Natural language interaction for financial forecasting
- **LangChain Integration**: Powered by OpenAI for intelligent responses
- **Context Awareness**: Remembers conversation history and preferences

## 🚀 **Quick Start**

### **1. Basic Forecast (Minimal Input)**
```bash
curl -X POST "https://your-api-url/guided_forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "My Company",
    "current_arr": 2800000,
    "net_new_arr": 800000,
    "enhanced_mode": false
  }'
```

### **2. Enhanced Forecast (Detailed Input)**
```bash
curl -X POST "https://your-api-url/guided_forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "My Company",
    "current_arr": 2800000,
    "net_new_arr": 800000,
    "enhanced_mode": true,
    "sector": "Cyber Security",
    "country": "United States",
    "currency": "USD"
  }'
```

### **3. Full Enhanced Forecast (Historical + Advanced)**
```bash
curl -X POST "https://your-api-url/guided_forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "My Company",
    "current_arr": 2800000,
    "net_new_arr": 800000,
    "enhanced_mode": true,
    "sector": "Data & Analytics",
    "country": "Israel",
    "currency": "USD",
    "historical_arr": {
      "q1_arr": 1000000,
      "q2_arr": 1400000,
      "q3_arr": 2000000,
      "q4_arr": 2800000
    },
    "advanced_mode": true,
    "advanced_metrics": {
      "magic_number": 0.95,
      "gross_margin": 82.0,
      "headcount": 70
    }
  }'
```

## 📊 **Available Options**

### **🏢 Sectors (7 options)**
- Cyber Security (19.4% of training data)
- Data & Analytics (14.5%)
- Infrastructure & Network (12.4%)
- Communication & Collaboration (10.8%)
- Marketing & Customer Experience (9.1%)
- FinTech (7.8%)
- Sales & Productivity (7.1%)
- Other (18.9%)

### **🌍 Countries (4 options)**
- United States (55.5% of training data)
- Israel (14.3%)
- Germany (10.3%)
- United Kingdom (6.4%)
- Other (13.5%)

### **💰 Currencies (5 options)**
- USD (87.7% of training data)
- EUR (10.5%)
- GBP (1.6%)
- CAD (0.1%)
- Other (0.1%)

## 🔗 **API Endpoints**

### **Core Endpoints**
- `GET /` - API documentation and available options
- `POST /guided_forecast` - Main forecasting endpoint
- `POST /chat` - Conversational AI interface
- `POST /predict_csv` - CSV upload for batch predictions
- `GET /makro-analysis` - Macroeconomic indicators

### **Response Format**
```json
{
  "company_name": "My Company",
  "forecast_results": [
    {
      "Future Quarter": "Q1 2025",
      "Realistic": 100.7,
      "Pessimistic": 90.6,
      "Optimistic": 110.7,
      "Realistic_ARR": 3080000,
      "Pessimistic_ARR": 2770000,
      "Optimistic_ARR": 3390000
    }
  ],
  "model_used": "LightGBM Model with Uncertainty (±10%)",
  "forecast_success": true,
  "insights": {
    "size_category": "Growth Stage",
    "growth_insight": "Growth rate: 40.0%",
    "efficiency_insight": "Magic Number: 0.80"
  }
}
```

## 🛠️ **Technical Requirements**

- **Python**: 3.10.18+
- **Dependencies**: See `requirements.txt`
- **Model**: LightGBM trained on 5,085+ company quarters
- **Data**: Real SaaS company financial data

## 🚀 **Deployment**

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn fastapi_app:app --reload
```

### **Production Deployment**
- **Render**: Configured with `render.yaml`
- **Environment Variables**: Set `OPENAI_API_KEY` for chat functionality
- **CORS**: Configured for web frontend integration

## 📈 **Model Performance**

- **Training Data**: 5,085+ company quarters
- **Features**: 158+ engineered features
- **Algorithm**: LightGBM gradient boosting
- **Uncertainty**: ±10% confidence bands
- **Fallback**: Robust error handling with alternative calculations

## 🔒 **Input Validation**

- **Bulletproof Validation**: Pydantic models ensure data integrity
- **Clear Error Messages**: Helpful feedback for invalid inputs
- **Flexible Inputs**: Works with minimal or detailed information
- **Data-Driven Defaults**: Intelligent fallbacks based on training data

## 📚 **Documentation**

- **Auto-Generated Docs**: Available at `/docs` when running
- **OpenAPI Schema**: Complete API specification
- **Example Requests**: Provided in documentation
- **Response Schemas**: All endpoints documented

## 🤝 **Integration**

Perfect for integration with:
- **Lovable**: Production-ready API for no-code platforms
- **Web Applications**: CORS configured for frontend integration
- **Mobile Apps**: RESTful API with JSON responses
- **Data Pipelines**: CSV upload and batch processing support

---

## 🎉 **Ready for Production**

This API is fully prepared for production deployment with:
- ✅ Bulletproof input validation
- ✅ Data-driven categorical options
- ✅ Comprehensive error handling
- ✅ Production-ready deployment config
- ✅ Complete API documentation
- ✅ Flexible input modes (basic/enhanced)
- ✅ Uncertainty quantification
- ✅ Historical data support 