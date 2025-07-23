import os
import pandas as pd
import requests
from datetime import datetime, timedelta

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
CACHE_FILE = "gpr_data_cache.xlsx"
CACHE_TIMESTAMP = "gpr_cache_timestamp.txt"
CACHE_DAYS = 31  # Update once a month


def fetch_and_cache_gpr():
    """Download and cache the GPR Excel file if cache is older than CACHE_DAYS."""
    now = datetime.utcnow()
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_TIMESTAMP):
        with open(CACHE_TIMESTAMP, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        if (now - last_fetch).days < CACHE_DAYS:
            return CACHE_FILE  # Use cached file
    # Download and cache
    r = requests.get(GPR_URL)
    with open(CACHE_FILE, 'wb') as f:
        f.write(r.content)
    with open(CACHE_TIMESTAMP, 'w') as f:
        f.write(now.isoformat())
    return CACHE_FILE


def load_gpr_dataframe():
    """Load the GPR data as a pandas DataFrame, with a datetime index and GPR column."""
    cache_path = fetch_and_cache_gpr()
    df = pd.read_excel(cache_path, skiprows=1)
    # Expect columns: 'Year', 'Month', 'GPR'
    df = df[['Year', 'Month', 'GPR']].dropna()
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.set_index('Date').sort_index()
    return df[['GPR']]


def gpr_traffic_light_analysis():
    """
    Return dict with last 12 months, latest z-score, traffic light, and opinion.
    """
    df = load_gpr_dataframe()
    last_10_years = df[-120:]
    last_12_months = df[-12:]
    mu = last_10_years['GPR'].mean()
    sigma = last_10_years['GPR'].std()
    latest_value = last_12_months['GPR'][-1]
    z = (latest_value - mu) / sigma if sigma != 0 else 0
    # Traffic light rule
    if z < -1:
        light = 'green'
        opinion = "Geopolitical risk is low right now."
    elif z > 1:
        light = 'red'
        opinion = "Geopolitical risk is high right now."
    else:
        light = 'orange'
        opinion = "Geopolitical risk is moderate right now."
    return {
        "last_12_months": last_12_months['GPR'].round(2).tolist(),
        "latest_z_score": round(z, 2),
        "traffic_light": light,
        "opinion": opinion
    } 