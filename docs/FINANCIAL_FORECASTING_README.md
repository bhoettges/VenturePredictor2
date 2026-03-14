# Financial Forecasting System

A comprehensive financial forecasting system using LightGBM for multi-step ARR (Annual Recurring Revenue) predictions.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

1. **Update the dataset path** in `train_model.py`:
   ```python
   filepath = "your_dataset_path.csv"  # ← Update this to your actual dataset path
   ```

2. **Run the training script**:
   ```bash
   python train_model.py
   ```

3. **Expected output**:
   - Model saved as `lightgbm_financial_model.pkl`
   - Performance visualizations saved as PNG files
   - Console output showing R² scores and feature importance

### 3. Make Predictions

```bash
python financial_prediction.py
```

This will load the trained model and make predictions on sample data.

## 📁 File Structure

```
FYP/
├── financial_forecasting_model.py    # Main training script with full algorithm
├── financial_prediction.py           # Prediction script for new data
├── train_model.py                   # Simple training wrapper
├── lightgbm_financial_model.pkl     # Trained model (created after training)
├── model_performance.png            # Performance visualization
├── feature_importance.png           # Feature importance plot
└── requirements.txt                 # Dependencies
```

## 🔧 How It Works

### Data Processing Pipeline

1. **Data Loading & Cleaning**:
   - Converts quarters to sortable time indices
   - Applies nuanced imputation (forward-fill for stock variables, 0 for flow variables)
   - Handles missing values with company-specific medians

2. **Feature Engineering**:
   - **Temporal Features**: Lags (1, 2, 4 quarters) and rolling statistics
   - **SaaS Metrics**: Magic Number, Burn Multiple, ARR per Headcount
   - **Growth Metrics**: Quarter-over-quarter growth rates
   - **Outlier Handling**: Winsorization to 1st and 99th percentiles

3. **Multi-Step Targets**:
   - Creates 4-quarter ahead forecasts (Target_Q1, Target_Q2, Target_Q3, Target_Q4)
   - Uses ARR YoY Growth as the target variable

### Model Architecture

- **Algorithm**: LightGBM with MAE objective (robust to outliers)
- **Pipeline**: StandardScaler → MultiOutputRegressor(LightGBM)
- **Validation**: Temporal split by company (80% train, 20% test)
- **Features**: 100+ engineered features including lags, ratios, and rolling statistics

## 📊 Expected Dataset Format

Your CSV file should contain these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `Financial Quarter` | Quarter identifier | "FY24 Q1" |
| `id_company` | Company identifier | "Company A" |
| `cARR` | Contracted Annual Recurring Revenue | 10000000 |
| `ARR YoY Growth (in %)` | Year-over-year growth | 0.70 |
| `Revenue YoY Growth (in %)` | Revenue growth rate | 0.65 |
| `Gross Margin (in %)` | Gross margin percentage | 80 |
| `Sales & Marketing` | S&M spend | 1000000 |
| `Cash Burn (OCF & ICF)` | Cash burn (negative) | -2000000 |
| `Headcount (HC)` | Employee count | 80 |
| `Customers (EoP)` | End-of-period customers | 200 |
| `Expansion & Upsell` | Expansion revenue | 600000 |
| `Churn & Reduction` | Churn (negative) | -100000 |

## 🎯 Making Predictions on New Data

### For a New Company

```python
from financial_prediction import load_trained_model, predict_future_arr

# Load the trained model
model_data = load_trained_model('lightgbm_financial_model.pkl')

# Prepare your company data (same format as training data)
new_company_data = pd.DataFrame({
    "id_company": ["Your Company"] * 8,
    "Financial Quarter": ["FY24 Q1", "FY24 Q2", ...],
    "cARR": [10000000, 11500000, ...],
    # ... other columns
})

# Make predictions
forecast = predict_future_arr(model_data, new_company_data)
print(forecast)
```

### Expected Output

```
--- 🔮 FORECAST FOR Your Company ---
Future Quarter  Predicted YoY Growth (%)  Predicted Absolute cARR (€)
FY26 Q1                             65.2%                   41,300,000
FY26 Q2                             63.8%                   44,200,000
FY26 Q3                             62.1%                   47,100,000
FY26 Q4                             60.5%                   50,000,000
```

## 📈 Model Performance

The model typically achieves:
- **Overall R²**: 0.60-0.80 (depending on data quality)
- **Individual Quarter R²**: 0.55-0.75
- **MAE**: 0.05-0.15 (5-15 percentage points)

## 🔍 Key Features

### Advanced SaaS Metrics

1. **Magic Number**: New ARR generated per $1 of S&M spend
2. **Burn Multiple**: Cash burned to generate $1 of new ARR
3. **ARR per Headcount**: Revenue efficiency metric

### Temporal Features

1. **Lags**: 1, 2, and 4-quarter historical values
2. **Rolling Statistics**: 4-quarter mean and standard deviation
3. **Growth Rates**: Quarter-over-quarter changes

### Robust Handling

1. **Outlier Management**: Winsorization prevents extreme values
2. **Missing Data**: Intelligent imputation strategies
3. **Data Leakage Prevention**: All features use only past information

## 🛠️ Customization

### Modify Feature Engineering

Edit the `engineer_features()` function in both training and prediction scripts to add new features.

### Adjust Model Parameters

In `financial_forecasting_model.py`, modify the LightGBM parameters:

```python
lgbm = lgb.LGBMRegressor(
    objective='regression_l1', 
    n_estimators=1000,        # Increase for better performance
    learning_rate=0.05,        # Lower for more stable training
    num_leaves=31,             # Adjust tree complexity
    # ... other parameters
)
```

### Change Forecast Horizon

Modify the `horizon` parameter in `create_multistep_targets()` to predict more or fewer quarters ahead.

## 🚨 Troubleshooting

### Common Issues

1. **"Dataset file not found"**
   - Update the `filepath` variable in `train_model.py`
   - Ensure your CSV file exists and is readable

2. **"Model file not found"**
   - Run `train_model.py` first to create the model
   - Check that `lightgbm_financial_model.pkl` exists

3. **Memory errors**
   - Reduce `n_estimators` in LightGBM parameters
   - Use a smaller subset of your dataset for testing

4. **Poor performance**
   - Check data quality and completeness
   - Ensure sufficient historical data (at least 8 quarters per company)
   - Verify that growth rates are calculated correctly

### Data Quality Checklist

- [ ] At least 8 quarters of data per company
- [ ] Consistent quarter naming (e.g., "FY24 Q1")
- [ ] No extreme outliers in financial metrics
- [ ] Missing values handled appropriately
- [ ] Growth rates calculated correctly

## 📚 Advanced Usage

### Integration with FastAPI

You can integrate this model into your existing FastAPI application:

```python
from fastapi import FastAPI
from financial_prediction import load_trained_model, predict_future_arr

app = FastAPI()
model_data = load_trained_model()

@app.post("/predict")
async def predict_arr(company_data: dict):
    df = pd.DataFrame(company_data)
    forecast = predict_future_arr(model_data, df)
    return forecast.to_dict('records')
```

### Batch Predictions

For multiple companies:

```python
def predict_multiple_companies(model_data, companies_data):
    results = {}
    for company_id, company_df in companies_data.items():
        forecast = predict_future_arr(model_data, company_df)
        results[company_id] = forecast
    return results
```

## 🤝 Contributing

To improve the model:

1. **Feature Engineering**: Add domain-specific features
2. **Model Selection**: Try different algorithms (XGBoost, CatBoost)
3. **Ensemble Methods**: Combine multiple models
4. **Hyperparameter Tuning**: Use Optuna or similar tools

## 📄 License

This project is part of your FYP (Final Year Project). Use responsibly and cite appropriately. 