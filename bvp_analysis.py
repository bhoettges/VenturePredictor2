import os
import pandas as pd
import requests
from datetime import datetime
import matplotlib.pyplot as plt

BVP_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQEMCLOUDN"
CACHE_FILE = "bvp_data_cache.csv"
CACHE_TIMESTAMP = "bvp_cache_timestamp.txt"
CACHE_DAYS = 7  # Update weekly


def fetch_and_cache_bvp():
    now = datetime.utcnow()
    if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_TIMESTAMP):
        with open(CACHE_TIMESTAMP, 'r') as f:
            last_fetch = datetime.fromisoformat(f.read().strip())
        if (now - last_fetch).days < CACHE_DAYS:
            return CACHE_FILE
    r = requests.get(BVP_URL)
    with open(CACHE_FILE, 'wb') as f:
        f.write(r.content)
    with open(CACHE_TIMESTAMP, 'w') as f:
        f.write(now.isoformat())
    return CACHE_FILE


def load_bvp_dataframe():
    cache_path = fetch_and_cache_bvp()
    df = pd.read_csv(cache_path)
    df = df.rename(columns={"observation_date": "date", "NASDAQEMCLOUDN": "bvp"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["bvp"] = pd.to_numeric(df["bvp"], errors='coerce')
    df = df.dropna(subset=["bvp"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df[["bvp"]]

def bvp_trend_analysis():
    df = load_bvp_dataframe()
    
    min_val = df['bvp'].min()
    max_val = df['bvp'].max()
    
    green_threshold = min_val + (max_val - min_val) * 0.33
    red_threshold = min_val + (max_val - min_val) * 0.66

    df_monthly = df.resample('ME').mean()
    
    end_date = df_monthly.index.max()
    start_date = end_date - pd.DateOffset(months=11)
    last_12 = df_monthly[(df_monthly.index >= start_date) & (df_monthly.index <= end_date)]
    
    if last_12.empty:
        return {
            "last_12_months_bvp": [],
            "start_value": 0,
            "end_value": 0,
            "change": 0,
            "traffic_light": "grey",
            "opinion": "Not enough data for a 12-month trend analysis."
        }
        
    values = last_12["bvp"].round(2).tolist()
    first = values[0]
    last_val = values[-1]
    diff = last_val - first

    if last_val >= red_threshold:
        light = 'red'
        opinion = (
            f"Cloud Index Valuation: High (> {red_threshold:.2f})\n\n"
            "The BVP Cloud Index is at a high valuation, suggesting the market for cloud software stocks may be overheated. "
            "This can indicate high investor optimism, but also carries a risk of a sharp correction.\n\n"
            "Investment Implications:\n"
            "• Public market valuations are high, potentially driving up private valuations\n"
            "• Favorable environment for IPOs and exits\n"
            "• Increased competition for deals as more capital flows into the sector\n"
            "• Risk of valuation compression if public markets correct\n\n"
            "Recommended Actions:\n"
            "• Exercise caution with new investments at high valuations\n"
            "• Consider taking some money off the table for mature portfolio companies\n"
            "• Focus on companies with strong fundamentals and sustainable growth"
        )
    elif last_val <= green_threshold:
        light = 'green'
        opinion = (
            f"Cloud Index Valuation: Low (< {green_threshold:.2f})\n\n"
            "The BVP Cloud Index is at a low valuation, which may present a buying opportunity for investors. "
            "This could be due to a broader market downturn or specific concerns about the cloud sector.\n\n"
            "Investment Implications:\n"
            "• Lower entry valuations for new investments\n"
            "• Potential for significant upside as the market recovers\n"
            "• More challenging environment for fundraising and exits\n"
            "• Less competition for deals\n\n"
            "Recommended Actions:\n"
            "• Increase investment pace to capitalize on lower valuations\n"
            "• Support portfolio companies in extending their runway\n"
            "• Focus on companies with strong product-market fit and a clear path to profitability"
        )
    else:
        light = 'yellow'
        opinion = (
            f"Cloud Index Valuation: Moderate ({green_threshold:.2f} - {red_threshold:.2f})\n\n"
            "The BVP Cloud Index is in a moderate range, indicating a balanced market for cloud software stocks. "
            "This environment supports steady growth and investment without the extremes of overheating or a downturn.\n\n"
            "Investment Implications:\n"
            "• Stable valuation environment for both new and existing investments\n"
            "• Steady deal flow and exit opportunities\n"
            "• A good balance between risk and reward\n\n"
            "Recommended Actions:\n"
            "• Maintain a disciplined investment approach\n"
            "• Focus on companies with strong competitive advantages\n"
            "• Prepare for both upside and downside scenarios"
        )
        
    return {
        "last_12_months_bvp": values,
        "start_value": round(first, 2),
        "end_value": round(last_val, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    }


if __name__ == "__main__":
    analysis_results = bvp_trend_analysis()
    print("BVP Cloud Index Trend Analysis:")
    print(f"  - Traffic Light: {analysis_results['traffic_light']}")
    print(f"  - Last 12 Months (Monthly Avg): {analysis_results['last_12_months_bvp']}")
    print(f"  - Start Value: {analysis_results['start_value']}")
    print(f"  - End Value: {analysis_results['end_value']}")
    print(f"  - Change: {analysis_results['change']}")
    print("\nOpinion:")
    print(analysis_results['opinion'])

    df = load_bvp_dataframe()
    plt.figure(figsize=(14, 7))
    plt.plot(df['bvp'], label='BVP Cloud Index (Close)')
    plt.title('BVP Nasdaq Emerging Cloud Index Performance')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.legend()
    plt.grid(True)
    plot_filename = "bvp_cloud_index_performance.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
