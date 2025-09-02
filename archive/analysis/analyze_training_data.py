#!/usr/bin/env python3
"""
Training Data Growth Rate Analysis
Analyzes the 202402_Copy.csv dataset to understand growth rate distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path: str = '202402_Copy.csv') -> pd.DataFrame:
    """Load and clean the training data."""
    print("📊 Loading training data...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {len(df)} rows from {file_path}")
        
        # Basic cleaning
        df = df.dropna(subset=['cARR', 'id_company'])
        df = df[df['cARR'] > 0]  # Remove negative/zero ARR
        
        # Convert quarters to sortable format
        df['Year'] = df['Financial Quarter'].str.extract(r'FY(\d{2,4})')[0].astype(int)
        df['Year'] = df['Year'].apply(lambda x: x + 2000 if x < 100 else x)
        df['Quarter Num'] = df['Financial Quarter'].str.extract(r'(Q[1-4])')[0].map({'Q1':1,'Q2':2,'Q3':3,'Q4':4})
        df['time_idx'] = df['Year'] * 4 + df['Quarter Num']
        
        # Sort by company and time
        df = df.sort_values(by=['id_company', 'time_idx']).reset_index(drop=True)
        
        print(f"✅ Cleaned data: {len(df)} rows from {df['id_company'].nunique()} companies")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def calculate_growth_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various growth rate metrics for each company."""
    print("📈 Calculating growth rates...")
    
    df_growth = df.copy()
    
    # Calculate Quarter-over-Quarter growth
    df_growth['QoQ_Growth'] = df_growth.groupby('id_company')['cARR'].pct_change(1) * 100
    
    # Calculate Year-over-Year growth (if we have enough data)
    df_growth['YoY_Growth'] = df_growth.groupby('id_company')['cARR'].pct_change(4) * 100
    
    # Calculate Net New ARR
    df_growth['Net_New_ARR'] = df_growth.groupby('id_company')['cARR'].diff()
    
    # Calculate Net New ARR as % of previous quarter ARR
    df_growth['Net_New_ARR_Pct'] = (df_growth['Net_New_ARR'] / df_growth.groupby('id_company')['cARR'].shift(1)) * 100
    
    # Clean infinite values
    df_growth = df_growth.replace([np.inf, -np.inf], np.nan)
    
    print("✅ Growth rates calculated")
    return df_growth

def analyze_growth_distribution(df_growth: pd.DataFrame) -> Dict:
    """Analyze the distribution of growth rates."""
    print("📊 Analyzing growth rate distribution...")
    
    # Get valid growth rates (non-null, finite)
    qoq_growth = df_growth['QoQ_Growth'].dropna()
    yoy_growth = df_growth['YoY_Growth'].dropna()
    net_new_pct = df_growth['Net_New_ARR_Pct'].dropna()
    
    analysis = {
        'QoQ_Growth': {
            'count': len(qoq_growth),
            'mean': qoq_growth.mean(),
            'median': qoq_growth.median(),
            'std': qoq_growth.std(),
            'min': qoq_growth.min(),
            'max': qoq_growth.max(),
            'q25': qoq_growth.quantile(0.25),
            'q75': qoq_growth.quantile(0.75),
            'q90': qoq_growth.quantile(0.90),
            'q95': qoq_growth.quantile(0.95),
            'q99': qoq_growth.quantile(0.99)
        },
        'YoY_Growth': {
            'count': len(yoy_growth),
            'mean': yoy_growth.mean(),
            'median': yoy_growth.median(),
            'std': yoy_growth.std(),
            'min': yoy_growth.min(),
            'max': yoy_growth.max(),
            'q25': yoy_growth.quantile(0.25),
            'q75': yoy_growth.quantile(0.75),
            'q90': yoy_growth.quantile(0.90),
            'q95': yoy_growth.quantile(0.95),
            'q99': yoy_growth.quantile(0.99)
        },
        'Net_New_ARR_Pct': {
            'count': len(net_new_pct),
            'mean': net_new_pct.mean(),
            'median': net_new_pct.median(),
            'std': net_new_pct.std(),
            'min': net_new_pct.min(),
            'max': net_new_pct.max(),
            'q25': net_new_pct.quantile(0.25),
            'q75': net_new_pct.quantile(0.75),
            'q90': net_new_pct.quantile(0.90),
            'q95': net_new_pct.quantile(0.95),
            'q99': net_new_pct.quantile(0.99)
        }
    }
    
    print("✅ Growth distribution analyzed")
    return analysis

def find_high_growth_examples(df_growth: pd.DataFrame, threshold: float = 20.0) -> pd.DataFrame:
    """Find examples of high-growth companies."""
    print(f"🔍 Finding companies with >{threshold}% quarterly growth...")
    
    high_growth = df_growth[df_growth['QoQ_Growth'] > threshold].copy()
    high_growth = high_growth.sort_values('QoQ_Growth', ascending=False)
    
    if len(high_growth) > 0:
        print(f"✅ Found {len(high_growth)} high-growth examples")
        return high_growth
    else:
        print(f"⚠️ No companies found with >{threshold}% quarterly growth")
        return pd.DataFrame()

def create_visualizations(df_growth: pd.DataFrame, analysis: Dict):
    """Create visualizations of the growth rate distributions."""
    print("📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Data Growth Rate Analysis', fontsize=16, fontweight='bold')
    
    # 1. QoQ Growth Distribution
    ax1 = axes[0, 0]
    qoq_data = df_growth['QoQ_Growth'].dropna()
    ax1.hist(qoq_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(analysis['QoQ_Growth']['mean'], color='red', linestyle='--', 
                label=f"Mean: {analysis['QoQ_Growth']['mean']:.2f}%")
    ax1.axvline(analysis['QoQ_Growth']['median'], color='orange', linestyle='--', 
                label=f"Median: {analysis['QoQ_Growth']['median']:.2f}%")
    ax1.axvline(40, color='green', linestyle='-', linewidth=2, 
                label="Your Company: 40%")
    ax1.set_xlabel('Quarter-over-Quarter Growth Rate (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('QoQ Growth Rate Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. YoY Growth Distribution
    ax2 = axes[0, 1]
    yoy_data = df_growth['YoY_Growth'].dropna()
    ax2.hist(yoy_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(analysis['YoY_Growth']['mean'], color='red', linestyle='--', 
                label=f"Mean: {analysis['YoY_Growth']['mean']:.2f}%")
    ax2.axvline(analysis['YoY_Growth']['median'], color='orange', linestyle='--', 
                label=f"Median: {analysis['YoY_Growth']['median']:.2f}%")
    ax2.set_xlabel('Year-over-Year Growth Rate (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('YoY Growth Rate Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Net New ARR % Distribution
    ax3 = axes[1, 0]
    net_new_data = df_growth['Net_New_ARR_Pct'].dropna()
    ax3.hist(net_new_data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.axvline(analysis['Net_New_ARR_Pct']['mean'], color='red', linestyle='--', 
                label=f"Mean: {analysis['Net_New_ARR_Pct']['mean']:.2f}%")
    ax3.axvline(analysis['Net_New_ARR_Pct']['median'], color='orange', linestyle='--', 
                label=f"Median: {analysis['Net_New_ARR_Pct']['median']:.2f}%")
    ax3.axvline(40, color='green', linestyle='-', linewidth=2, 
                label="Your Company: 40%")
    ax3.set_xlabel('Net New ARR as % of Previous Quarter ARR (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Net New ARR % Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Growth Rate Comparison
    ax4 = axes[1, 1]
    # Create box plots for comparison
    growth_data = [qoq_data, yoy_data, net_new_data]
    labels = ['QoQ Growth', 'YoY Growth', 'Net New ARR %']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    box_plot = ax4.boxplot(growth_data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Growth Rate (%)')
    ax4.set_title('Growth Rate Comparison (Box Plots)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_data_growth_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'training_data_growth_analysis.png'")
    plt.show()

def print_analysis_summary(analysis: Dict, df_growth: pd.DataFrame):
    """Print a comprehensive summary of the analysis."""
    print("\n" + "="*80)
    print("📊 TRAINING DATA GROWTH RATE ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🔍 QUARTER-OVER-QUARTER GROWTH RATES:")
    qoq = analysis['QoQ_Growth']
    print(f"  Count: {qoq['count']:,}")
    print(f"  Mean: {qoq['mean']:.2f}%")
    print(f"  Median: {qoq['median']:.2f}%")
    print(f"  25th Percentile: {qoq['q25']:.2f}%")
    print(f"  75th Percentile: {qoq['q75']:.2f}%")
    print(f"  90th Percentile: {qoq['q90']:.2f}%")
    print(f"  95th Percentile: {qoq['q95']:.2f}%")
    print(f"  99th Percentile: {qoq['q99']:.2f}%")
    print(f"  Range: {qoq['min']:.2f}% to {qoq['max']:.2f}%")
    
    print("\n🔍 YEAR-OVER-YEAR GROWTH RATES:")
    yoy = analysis['YoY_Growth']
    print(f"  Count: {yoy['count']:,}")
    print(f"  Mean: {yoy['mean']:.2f}%")
    print(f"  Median: {yoy['median']:.2f}%")
    print(f"  25th Percentile: {yoy['q25']:.2f}%")
    print(f"  75th Percentile: {yoy['q75']:.2f}%")
    print(f"  90th Percentile: {yoy['q90']:.2f}%")
    print(f"  95th Percentile: {yoy['q95']:.2f}%")
    print(f"  99th Percentile: {yoy['q99']:.2f}%")
    print(f"  Range: {yoy['min']:.2f}% to {yoy['max']:.2f}%")
    
    print("\n🔍 NET NEW ARR AS % OF PREVIOUS QUARTER:")
    net_new = analysis['Net_New_ARR_Pct']
    print(f"  Count: {net_new['count']:,}")
    print(f"  Mean: {net_new['mean']:.2f}%")
    print(f"  Median: {net_new['median']:.2f}%")
    print(f"  25th Percentile: {net_new['q25']:.2f}%")
    print(f"  75th Percentile: {net_new['q75']:.2f}%")
    print(f"  90th Percentile: {net_new['q90']:.2f}%")
    print(f"  95th Percentile: {net_new['q95']:.2f}%")
    print(f"  99th Percentile: {net_new['q99']:.2f}%")
    print(f"  Range: {net_new['min']:.2f}% to {net_new['max']:.2f}%")
    
    print("\n" + "="*80)
    print("🚨 KEY INSIGHTS:")
    print("="*80)
    
    # Calculate where your company falls
    your_qoq = 40.0
    your_yoy = ((1 + 0.40)**4 - 1) * 100  # Convert to annual
    
    qoq_percentile = (df_growth['QoQ_Growth'].dropna() < your_qoq).mean() * 100
    yoy_percentile = (df_growth['YoY_Growth'].dropna() < your_yoy).mean() * 100
    
    print(f"📈 Your Company Profile:")
    print(f"  Quarterly Growth: {your_qoq:.1f}%")
    print(f"  Annual Growth: {your_yoy:.1f}%")
    print(f"  QoQ Percentile: {qoq_percentile:.1f}% (Top {100-qoq_percentile:.1f}%)")
    print(f"  YoY Percentile: {yoy_percentile:.1f}% (Top {100-yoy_percentile:.1f}%)")
    
    print(f"\n⚠️  Data Bias Issues:")
    print(f"  • 75% of companies have QoQ growth < {qoq['q75']:.1f}%")
    print(f"  • 90% of companies have QoQ growth < {qoq['q90']:.1f}%")
    print(f"  • Your company's growth ({your_qoq:.1f}%) is in the top {100-qoq_percentile:.1f}%")
    print(f"  • The model learned from mostly low-growth companies")
    
    print(f"\n💡 Recommendations:")
    print(f"  • Filter training data to include more high-growth companies")
    print(f"  • Use fallback calculation for companies with >{qoq['q90']:.1f}% growth")
    print(f"  • Consider retraining with balanced growth rate distribution")

def main():
    """Main analysis function."""
    print("🚀 Training Data Growth Rate Analysis")
    print("="*50)
    
    # Load and clean data
    df = load_and_clean_data()
    if df is None:
        return
    
    # Calculate growth rates
    df_growth = calculate_growth_rates(df)
    
    # Analyze distribution
    analysis = analyze_growth_distribution(df_growth)
    
    # Find high-growth examples
    high_growth_examples = find_high_growth_examples(df_growth, threshold=20.0)
    
    # Print summary
    print_analysis_summary(analysis, df_growth)
    
    # Show high-growth examples if any
    if len(high_growth_examples) > 0:
        print(f"\n🔍 HIGH-GROWTH EXAMPLES (>20% QoQ):")
        print(high_growth_examples[['id_company', 'Financial Quarter', 'cARR', 'QoQ_Growth', 'Net_New_ARR']].head(10))
    
    # Create visualizations
    create_visualizations(df_growth, analysis)
    
    print("\n✅ Analysis complete! Check 'training_data_growth_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()
