# 🚀 Render Deployment Guide

## Prerequisites
- Render account (free tier available)
- Git repository with your code

## Deployment Steps

### 1. Push to Git
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### 2. Deploy on Render
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure:
   - **Name**: `financial-forecasting-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

### 3. Environment Variables
Add these in Render dashboard:
- `OPENAI_API_KEY`: Your OpenAI API key (if using chat features)

### 4. Deploy
Click "Create Web Service" and wait for build to complete.

## API Endpoints
- **Main**: `https://your-app.onrender.com/`
- **CSV Prediction**: `POST /predict_csv`
- **Chat**: `POST /chat`
- **Guided Forecast**: `POST /guided_forecast`

## Testing
Test with your CSV file:
```bash
curl -X POST "https://your-app.onrender.com/predict_csv" \
  -F "file=@test_company_2024.csv" \
  -F "model=lightgbm"
```

## Troubleshooting
- Check build logs in Render dashboard
- Ensure all dependencies are in requirements.txt
- Verify Python version compatibility
