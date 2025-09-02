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
    # Convert date column to datetime and ensure it's properly formatted
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.dropna(subset=["move"])
    df = df.sort_values("date")
    # Set the datetime column as index
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
            "Bond Market Volatility: High (MOVE >150)\n\n"
            "The MOVE index indicates elevated uncertainty in fixed income markets, suggesting significant "
            "volatility in interest rates driven by central bank policy uncertainty, inflation concerns, "
            "or credit market stress.\n\n"
            "Investment Implications:\n"
            "• Higher cost of capital across all funding sources\n"
            "• Extended fundraising cycles as LPs reassess allocations\n"
            "• Valuation compression due to higher discount rates\n"
            "• Reduced appetite for long-term growth investments\n"
            "• Increased focus on profitability over growth\n"
            "• Secondary market activity may increase for liquidity\n\n"
            "Recommended Actions:\n"
            "• Extend runway by 9-18 months beyond current projections\n"
            "• Focus on unit economics and path to profitability\n"
            "• Diversify funding sources and reduce debt dependency\n"
            "• Consider defensive positioning in recession-resistant sectors\n"
            "• Strengthen relationships with existing investors\n"
            "• Monitor for opportunistic acquisitions of rate-sensitive assets"
        )
    elif last_val <= 80:
        light = 'green'
        opinion = (
            "Bond Market Volatility: Low (MOVE <80)\n\n"
            "The MOVE index indicates stable and predictable interest rate conditions. "
            "This low volatility environment typically correlates with strong investor confidence "
            "and favorable conditions for capital deployment.\n\n"
            "Investment Implications:\n"
            "• Lower cost of capital and improved access to debt financing\n"
            "• Accelerated fundraising with improved terms\n"
            "• Higher valuations due to lower discount rates\n"
            "• Strong IPO environment with stable pricing\n"
            "• Increased LP appetite for growth investments\n"
            "• Favorable conditions for leveraged buyouts and M&A\n\n"
            "Recommended Actions:\n"
            "• Accelerate growth initiatives and market expansion\n"
            "• Pursue strategic M&A and leveraged opportunities\n"
            "• Consider earlier exit timing for mature companies\n"
            "• Increase investment pace in growth-stage deals\n"
            "• Leverage improved access to debt financing\n"
            "• Explore new markets and geographies"
        )
    else:
        light = 'yellow'
        opinion = (
            "Bond Market Volatility: Moderate (MOVE 80-150)\n\n"
            "The MOVE index is in a normal range, indicating balanced interest rate expectations. "
            "This environment supports steady deal flow with moderate risk-taking, "
            "though requires ongoing monitoring of monetary policy developments.\n\n"
            "Investment Implications:\n"
            "• Normalized cost of capital with reasonable terms\n"
            "• Stable valuations with slight upward bias\n"
            "• Steady deal flow across all stages\n"
            "• IPO windows available but timing-dependent\n"
            "• Moderate LP appetite with selective risk-taking\n\n"
            "Recommended Actions:\n"
            "• Maintain disciplined growth with measured risk-taking\n"
            "• Balance growth investments with portfolio resilience\n"
            "• Monitor monetary policy indicators for timing decisions\n"
            "• Prepare contingency plans for rate volatility\n"
            "• Focus on quality over quantity in deal flow"
        )
    return {
        "last_12_months_move": values,
        "start_value": round(first, 2),
        "end_value": round(last_val, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 