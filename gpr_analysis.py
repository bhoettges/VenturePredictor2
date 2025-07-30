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
    # Traffic light logic and detailed opinion
    if last >= 140:
        light = 'red'
        opinion = (
            "High GPR (> ≈ 140) – risk increased\n"
            "The index is sitting near 150 (June 2025), its highest level since the immediate aftermath of Russia’s 2022 invasion of Ukraine, as three live conflicts (Ukraine‑Russia, Israel‑Hamas‑Iran, and renewed Red‑Sea flashpoints) collide with an escalated U.S.–China tariff war.\n"
            "MacroMicro\nThe Guardian\nMorning Consult Pro\n\n"
            "VC implication: Global allocators are in “risk‑off” mode. Expect slower fundraising, tighter follow‑on rounds and a valuation haircut of 10‑20 % on late‑stage deals.\n\n"
            "Tactical edge: Capital is flowing toward defence tech, cyber‑security, energy‑security hardware and AI‑enabled resilience platforms. Keep dry‑powder for bridge rounds and lengthen due‑diligence cycles."
        )
    elif last <= 80:
        light = 'green'
        opinion = (
            "Low GPR (< ≈ 80) – risk decreased\n"
            "A sub‑80 reading means press mentions of geopolitical tension are at least 20 % below the 2000‑09 benchmark, indicating a lull in international flashpoints and a more predictable policy backdrop.\n"
            "federalreserve.gov\n\n"
            "VC implication: Risk‑on sentiment returns; cross‑border M&A and IPO windows reopen, pushing growth‑stage valuations up and shortening time‑to‑term‑sheet for seed deals.\n\n"
            "Tactical edge: Lean into expansion‑stage bets, accelerate hiring in go‑to‑market roles, and revisit markets previously deemed “too geopolitically risky.”"
        )
    elif 90 <= last <= 120:
        light = 'yellow'
        opinion = (
            "Moderate GPR (≈ 90 – 120) – risk stable\n"
            "The index is hovering around its long‑run mean of 100—by construction the average level during 2000‑09—signalling that geopolitical headlines are no longer intensifying but remain a constant drumbeat.\n"
            "federalreserve.gov\n\n"
            "VC implication: LP appetite is cautious‑but‑steady; deal volumes and pricing mirror 12‑month averages.\n\n"
            "Tactical edge: Focus on capital‑efficient growth, diversify supply chains early, and build optionality for both strategic and IPO exits in 2026‑27."
        )
    else:
        # For values between 80 and 90, or 120 and 140, interpolate with a generic message
        if last < 90:
            light = 'green'
            opinion = (
                "GPR is below average, indicating a period of reduced geopolitical tension. VC sentiment is improving, but not yet fully risk-on."
            )
        else:
            light = 'red'
            opinion = (
                "GPR is above average, indicating heightened but not extreme geopolitical risk. VC activity may be cautious, with selective risk-taking."
            )
    return {
        "last_12_months_gprh": values,
        "start_value": round(first, 2),
        "end_value": round(last, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 