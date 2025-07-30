import os
import pandas as pd
from datetime import datetime
import yfinance as yf

CACHE_FILE = "MOVE.csv"
CACHE_TIMESTAMP = "move_cache_timestamp.txt"
CACHE_DAYS = 7  # Update weekly


def fetch_and_cache_move():
    now = datetime.utcnow()
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_TIMESTAMP):
        with open(CACHE_TIMESTAMP, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        if (now - last_fetch).days < CACHE_DAYS:
            return CACHE_FILE
    move = (
        yf.Ticker("^MOVE")
          .history(period="max")
          .loc[:, ["Close"]]
          .rename(columns={"Close": "MOVE"})
    )
    move.to_csv(CACHE_FILE)
    with open(CACHE_TIMESTAMP, 'w') as f:
        f.write(now.isoformat())
    return CACHE_FILE


def load_move_dataframe():
    cache_path = fetch_and_cache_move()
    df = pd.read_csv(cache_path)
    # Rename columns first, then parse dates
    df = df.rename(columns={"Date": "date", "MOVE": "move"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["move"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df[["move"]]


def move_trend_analysis():
    df = load_move_dataframe()
    # Use last 12 months (resample to monthly mean)
    df_monthly = df.resample('ME').mean()
    
    # Get the actual last 12 months, not just last 12 rows
    end_date = df_monthly.index.max()
    start_date = end_date - pd.DateOffset(months=11)  # 12 months total
    last_12 = df_monthly[(df_monthly.index >= start_date) & (df_monthly.index <= end_date)]
    
    values = last_12["move"].round(2).tolist()
    first = values[0]
    last_val = values[-1]
    diff = last_val - first
    # Traffic light logic and VC/investing opinion
    if last_val >= 150:
        light = 'red'
        opinion = (
            "High MOVE (>150) – Bond market volatility increased\n"
            "The MOVE index is above 150, indicating significant uncertainty in fixed income markets.\n"
            "VC implication: Higher cost of capital, slower fundraising, and increased caution in late-stage deals.\n"
            "Tactical edge: Focus on capital efficiency, extend runway, and monitor interest rate risk."
        )
    elif last_val <= 80:
        light = 'green'
        opinion = (
            "Low MOVE (<80) – Bond market volatility decreased\n"
            "The MOVE index is below 80, reflecting stable rates and predictable funding conditions.\n"
            "VC implication: Easier access to capital, improved exit environment, and higher risk appetite.\n"
            "Tactical edge: Accelerate growth, pursue opportunistic financing, and consider expansion."
        )
    else:
        light = 'yellow'
        opinion = (
            "Moderate MOVE (80–150) – Bond market volatility stable\n"
            "The MOVE index is in a normal range, suggesting balanced risk in fixed income markets.\n"
            "VC implication: Steady deal flow, moderate valuations, and selective risk-taking.\n"
            "Tactical edge: Maintain discipline, diversify funding sources, and monitor macro trends."
        )
    return {
        "last_12_months_move": values,
        "start_value": round(first, 2),
        "end_value": round(last_val, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 