# Predictive Benchmarking Tool

This project predicts ARR YoY growth for companies using machine learning models trained on financial data.

## Features
- Data loading and preprocessing from Excel
- Time-series and single-output ML models (XGBoost, Random Forest, SVR)
- Visualization of feature importance, predictions, and residuals
- **NEW:** Conversational chat agent powered by LangChain and OpenAI for interactive predictions via a web API

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Usage

### 1. Train the Model
Run the training script to train and save your ML model:
```bash
python3 main.py
```

### 2. Run the FastAPI Backend
Start the FastAPI server (locally or on Render):
```bash
uvicorn fastapi_app:app --reload
```

This will expose endpoints for predictions and a chat endpoint powered by LangChain and OpenAI.

You can then build a web frontend (e.g., React) that interacts with the `/chat` endpoint for a conversational experience.

## Data
- Place your Excel data file (e.g., `202402_Copy.xlsx`) in the project directory.

## Visualization
- Output plots are saved in the `output/` directory.

## Deployment
- Deploy the FastAPI backend to [Render](https://render.com/) or another cloud provider for production use.
- (Optional) Deploy your frontend separately (e.g., Vercel, Netlify).

## Notes
- The previous CLI agent and Docker setup have been removed in favor of a web-based API workflow. 