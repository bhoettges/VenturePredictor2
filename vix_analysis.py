import os
import pandas as pd
import requests
from datetime import datetime

VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
CACHE_FILE = "vix_data_cache.csv"
CACHE_TIMESTAMP = "vix_cache_timestamp.txt"
CACHE_DAYS = 7  # Update weekly


def fetch_and_cache_vix():
    now = datetime.utcnow()
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_TIMESTAMP):
        with open(CACHE_TIMESTAMP, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        if (now - last_fetch).days < CACHE_DAYS:
            return CACHE_FILE
    r = requests.get(VIX_URL)
    with open(CACHE_FILE, 'wb') as f:
        f.write(r.content)
    with open(CACHE_TIMESTAMP, 'w') as f:
        f.write(now.isoformat())
    return CACHE_FILE


def load_vix_dataframe():
    cache_path = fetch_and_cache_vix()
    df = pd.read_csv(cache_path)
    # FRED VIX CSV: columns are observation_date, VIXCLS
    # Rename columns first, then parse dates
    df = df.rename(columns={"observation_date": "date", "VIXCLS": "vix"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["vix"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df[["vix"]]


def vix_trend_analysis():
    df = load_vix_dataframe()
    # Use last 12 months (resample to monthly mean)
    df_monthly = df.resample('ME').mean()
    
    # Get the actual last 12 months, not just last 12 rows
    end_date = df_monthly.index.max()
    start_date = end_date - pd.DateOffset(months=11)  # 12 months total
    last_12 = df_monthly[(df_monthly.index >= start_date) & (df_monthly.index <= end_date)]
    
    values = last_12["vix"].round(2).tolist()
    first = values[0]
    last_val = values[-1]
    diff = last_val - first
    # Traffic light logic and VC/investing opinion
    if last_val >= 30:
        light = 'red'
        opinion = (
            "High VIX (>30) – Volatility increased\n"
            "The VIX index is above 30, indicating heightened market uncertainty and risk aversion.\n"
            "VC implication: Fundraising slows, valuations compress, and exits become more challenging.\n"
            "Tactical edge: Focus on capital efficiency, extend runway, and prioritize capital-efficient growth."
        )
    elif last_val <= 15:
        light = 'green'
        opinion = (
            "Low VIX (<15) – Volatility decreased\n"
            "The VIX index is below 15, reflecting calm markets and investor confidence.\n"
            "VC implication: Risk-on sentiment, easier fundraising, and higher valuations.\n"
            "Tactical edge: Accelerate growth plans, pursue opportunistic M&A, and consider IPO windows."
        )
    else:
        light = 'yellow'
        opinion = (
            "Moderate VIX (15–30) – Volatility stable\n"
            "The VIX index is in a normal range, suggesting balanced risk appetite.\n"
            "VC implication: Steady deal flow, reasonable pricing, and selective risk-taking.\n"
            "Tactical edge: Maintain discipline, diversify bets, and monitor macro signals."
        )
    return {
        "last_12_months_vix": values,
        "start_value": round(first, 2),
        "end_value": round(last_val, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 