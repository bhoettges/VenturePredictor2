import os
import pandas as pd
import requests
from datetime import datetime

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
CACHE_FILE = "gpr_data_cache.xlsx"
CACHE_TIMESTAMP = "gpr_cache_timestamp.txt"
CACHE_DAYS = 31  # Update once a month


def fetch_and_cache_gpr():
    now = datetime.utcnow()
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_TIMESTAMP):
        with open(CACHE_TIMESTAMP, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        if (now - last_fetch).days < CACHE_DAYS:
            return CACHE_FILE
    r = requests.get(GPR_URL)
    with open(CACHE_FILE, 'wb') as f:
        f.write(r.content)
    with open(CACHE_TIMESTAMP, 'w') as f:
        f.write(now.isoformat())
    return CACHE_FILE


def load_gprh_dataframe():
    cache_path = fetch_and_cache_gpr()
    # Read the correct columns and parse 'month' as datetime
    df = pd.read_excel(
        cache_path,
        usecols=["month", "GPRH"],
        parse_dates=["month"]
    )
    df = df.dropna(subset=["month", "GPRH"])
    df = df.sort_values("month")
    df = df.set_index("month")
    return df[["GPRH"]]


def gprh_trend_analysis():
    df = load_gprh_dataframe()
    last_12 = df[-12:]
    values = last_12['GPRH'].round(2).tolist()
    first = values[0]
    last = values[-1]
    diff = last - first
    # Traffic light logic
    if abs(diff) <= 2.5:
        light = 'yellow'
        opinion = "Geopolitical risk is stable over the last year."
    elif last < first:
        light = 'green'
        opinion = "Geopolitical risk has decreased over the last year."
    else:
        light = 'red'
        opinion = "Geopolitical risk has increased over the last year."
    return {
        "last_12_months_gprh": values,
        "start_value": round(first, 2),
        "end_value": round(last, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 