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
    # Convert date column to datetime and ensure it's properly formatted
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.dropna(subset=["vix"])
    df = df.sort_values("date")
    # Set the datetime column as index
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
            "🚨 HIGH MARKET VOLATILITY (>30) - Fear Gauge Elevated\n\n"
            "The VIX index has surged above 30, indicating extreme market fear and uncertainty. "
            "This 'fear gauge' suggests investors are pricing in significant downside risk, "
            "often triggered by economic uncertainty, policy changes, or systemic concerns.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Public market volatility directly impacts private valuations\n"
            "• Fundraising becomes more challenging with extended timelines\n"
            "• Down rounds become more common across all stages\n"
            "• Exit valuations face 20-40% compression\n"
            "• LPs become more conservative and demand higher returns\n"
            "• Secondary market activity may increase as investors seek liquidity\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Extend runway by 6-12 months beyond current projections\n"
            "• Focus on unit economics and path to profitability\n"
            "• Consider bridge financing to weather the storm\n"
            "• Diversify revenue streams and customer concentration\n"
            "• Strengthen relationships with existing investors\n"
            "• Monitor for opportunistic acquisitions of distressed assets"
        )
    elif last_val <= 15:
        light = 'green'
        opinion = (
            "🟢 LOW MARKET VOLATILITY (<15) - Calm Waters Ahead\n\n"
            "The VIX index is below 15, indicating market complacency and investor confidence. "
            "This low volatility environment typically correlates with strong risk appetite "
            "and favorable conditions for capital deployment.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Accelerated fundraising with improved terms and valuations\n"
            "• Strong IPO and M&A exit environment\n"
            "• Increased LP appetite for risk and higher allocations\n"
            "• Favorable conditions for growth-stage investments\n"
            "• Reduced due diligence timelines\n"
            "• Higher multiples across all stages\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Accelerate growth initiatives and market expansion\n"
            "• Pursue strategic M&A opportunities\n"
            "• Consider earlier exit timing for mature companies\n"
            "• Increase investment pace in growth-stage deals\n"
            "• Leverage improved access to international capital\n"
            "• Explore new markets and geographies"
        )
    else:
        light = 'yellow'
        opinion = (
            "🟡 MODERATE MARKET VOLATILITY (15-30) - Balanced Environment\n\n"
            "The VIX index is in a normal range, indicating balanced market sentiment. "
            "This environment supports steady deal flow with moderate risk-taking, "
            "though requires ongoing monitoring of market conditions.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Normalized fundraising timelines with reasonable terms\n"
            "• Stable valuations with slight upward bias\n"
            "• Steady deal flow across all stages\n"
            "• IPO windows available but timing-dependent\n"
            "• Moderate LP appetite with selective risk-taking\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Maintain disciplined growth with measured risk-taking\n"
            "• Balance growth investments with portfolio resilience\n"
            "• Monitor market indicators for timing decisions\n"
            "• Prepare contingency plans for volatility spikes\n"
            "• Focus on quality over quantity in deal flow"
        )
    return {
        "last_12_months_vix": values,
        "start_value": round(first, 2),
        "end_value": round(last_val, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 