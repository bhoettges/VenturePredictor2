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
            "🚨 HIGH GEOPOLITICAL RISK (>140) - Critical Alert\n\n"
            "The GPRH index has surged to elevated levels, indicating severe geopolitical instability. "
            "This represents the highest risk environment since the immediate aftermath of Russia's 2022 invasion of Ukraine. "
            "Multiple concurrent crises are creating a perfect storm: Ukraine-Russia conflict escalation, "
            "Israel-Hamas-Iran tensions, Red Sea shipping disruptions, and intensified U.S.-China trade hostilities.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Fundraising cycles will extend 3-6 months longer than usual\n"
            "• Late-stage valuations face 15-25% compression\n"
            "• Cross-border M&A activity will significantly decline\n"
            "• IPO windows may close entirely for 6-12 months\n"
            "• LPs will demand higher risk premiums and stricter terms\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Prioritize runway extension over growth at all costs\n"
            "• Focus on capital efficiency and unit economics\n"
            "• Diversify supply chains away from geopolitical hotspots\n"
            "• Build strategic reserves for bridge financing\n"
            "• Consider defensive positioning in cybersecurity, energy security, and AI resilience\n"
            "• Lengthen due diligence cycles and add geopolitical risk assessments"
        )
    elif last <= 80:
        light = 'green'
        opinion = (
            "🟢 LOW GEOPOLITICAL RISK (<80) - Favorable Environment\n\n"
            "The GPRH index indicates a period of relative geopolitical calm, with press mentions of "
            "international tensions at least 20% below historical benchmarks. This suggests a more "
            "predictable policy environment and reduced uncertainty in global markets.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Accelerated fundraising cycles with improved terms\n"
            "• Higher valuations across all stages, especially growth\n"
            "• Resurgence in cross-border M&A and strategic exits\n"
            "• IPO windows reopening with strong investor appetite\n"
            "• Increased LP confidence and risk-on sentiment\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Accelerate growth initiatives and market expansion\n"
            "• Pursue opportunistic M&A and strategic partnerships\n"
            "• Consider earlier exit timing for mature portfolio companies\n"
            "• Increase investment pace in growth-stage opportunities\n"
            "• Revisit previously 'too risky' emerging markets\n"
            "• Leverage improved access to international capital"
        )
    elif 90 <= last <= 120:
        light = 'yellow'
        opinion = (
            "🟡 MODERATE GEOPOLITICAL RISK (90-120) - Stable but Watchful\n\n"
            "The GPRH index is hovering around its long-term average, indicating a balanced risk environment. "
            "While not crisis-level, geopolitical tensions remain a constant background factor that requires "
            "ongoing monitoring and strategic adaptation.\n\n"
            "🎯 VC IMPLICATIONS:\n"
            "• Normalized fundraising timelines with moderate terms\n"
            "• Stable valuations with slight upward bias\n"
            "• Steady deal flow with selective cross-border activity\n"
            "• IPO windows available but timing-dependent\n"
            "• Cautious but steady LP appetite\n\n"
            "💡 TACTICAL RESPONSE:\n"
            "• Maintain disciplined growth with measured risk-taking\n"
            "• Diversify geographic exposure and supply chains\n"
            "• Build optionality for both strategic and IPO exits\n"
            "• Monitor macro indicators for timing decisions\n"
            "• Balance growth investments with portfolio resilience\n"
            "• Prepare contingency plans for risk escalation"
        )
    else:
        # For values between 80 and 90, or 120 and 140, interpolate with a generic message
        if last < 90:
            light = 'green'
            opinion = (
                "🟢 BELOW-AVERAGE GEOPOLITICAL RISK - Improving Conditions\n\n"
                "The GPRH index is below historical averages, indicating improving geopolitical stability. "
                "While not yet fully risk-on, this environment supports cautious optimism and strategic growth initiatives.\n\n"
                "🎯 VC IMPLICATIONS:\n"
                "• Gradually improving fundraising conditions\n"
                "• Moderate valuation improvements expected\n"
                "• Selective expansion opportunities emerging\n\n"
                "💡 TACTICAL RESPONSE:\n"
                "• Begin strategic growth planning\n"
                "• Monitor for further risk reduction\n"
                "• Prepare for potential market expansion"
            )
        else:
            light = 'red'
            opinion = (
                "🟠 ELEVATED GEOPOLITICAL RISK - Caution Required\n\n"
                "The GPRH index is above average, indicating heightened but not extreme geopolitical tensions. "
                "This environment requires careful navigation and defensive positioning.\n\n"
                "🎯 VC IMPLICATIONS:\n"
                "• Extended fundraising timelines\n"
                "• Valuation pressure in certain sectors\n"
                "• Increased due diligence requirements\n\n"
                "💡 TACTICAL RESPONSE:\n"
                "• Prioritize runway and capital efficiency\n"
                "• Strengthen portfolio resilience\n"
                "• Monitor for further escalation"
            )
    return {
        "last_12_months_gprh": values,
        "start_value": round(first, 2),
        "end_value": round(last, 2),
        "change": round(diff, 2),
        "traffic_light": light,
        "opinion": opinion
    } 