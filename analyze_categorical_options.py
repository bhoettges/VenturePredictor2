#!/usr/bin/env python3
"""
Analyze categorical options for enhanced mode - determine optimal sector and country choices
"""

import pandas as pd
import numpy as np

def analyze_sector_options():
    """Analyze sector distribution and determine optimal options."""
    print("🔍 ANALYZING SECTOR OPTIONS")
    print("=" * 50)
    
    try:
        df = pd.read_csv('202402_Copy.csv')
        
        # Analyze sector distribution
        sector_counts = df['Sector'].value_counts()
        total_companies = len(df)
        
        print(f"📊 Total companies: {total_companies:,}")
        print(f"📊 Unique sectors: {len(sector_counts)}")
        print(f"\n📈 SECTOR DISTRIBUTION:")
        print("-" * 40)
        
        # Show all sectors with percentages
        for sector, count in sector_counts.items():
            percentage = (count / total_companies) * 100
            print(f"{sector:<35} {count:>6,} ({percentage:>5.1f}%)")
        
        # Determine optimal options (cover 80%+ of companies)
        cumulative_percentage = 0
        recommended_sectors = []
        
        print(f"\n🎯 RECOMMENDED SECTOR OPTIONS (Covering 80%+ of companies):")
        print("-" * 60)
        
        for sector, count in sector_counts.items():
            percentage = (count / total_companies) * 100
            cumulative_percentage += percentage
            recommended_sectors.append(sector)
            
            print(f"✅ {sector:<35} {count:>6,} ({percentage:>5.1f}%) | Cumulative: {cumulative_percentage:>5.1f}%")
            
            if cumulative_percentage >= 80:
                break
        
        # Add "Other" option for remaining sectors
        remaining_percentage = 100 - cumulative_percentage
        print(f"📝 Other sectors: {remaining_percentage:.1f}% of companies")
        
        return recommended_sectors, sector_counts
        
    except Exception as e:
        print(f"❌ Error analyzing sectors: {e}")
        return [], {}

def analyze_country_options():
    """Analyze country distribution and determine optimal options."""
    print(f"\n🌍 ANALYZING COUNTRY OPTIONS")
    print("=" * 50)
    
    try:
        df = pd.read_csv('202402_Copy.csv')
        
        # Analyze country distribution
        country_counts = df['Country'].value_counts()
        total_companies = len(df)
        
        print(f"📊 Total companies: {total_companies:,}")
        print(f"📊 Unique countries: {len(country_counts)}")
        print(f"\n📈 COUNTRY DISTRIBUTION:")
        print("-" * 40)
        
        # Show all countries with percentages
        for country, count in country_counts.items():
            percentage = (count / total_companies) * 100
            print(f"{country:<25} {count:>6,} ({percentage:>5.1f}%)")
        
        # Determine optimal options (cover 85%+ of companies)
        cumulative_percentage = 0
        recommended_countries = []
        
        print(f"\n🎯 RECOMMENDED COUNTRY OPTIONS (Covering 85%+ of companies):")
        print("-" * 60)
        
        for country, count in country_counts.items():
            percentage = (count / total_companies) * 100
            cumulative_percentage += percentage
            recommended_countries.append(country)
            
            print(f"✅ {country:<25} {count:>6,} ({percentage:>5.1f}%) | Cumulative: {cumulative_percentage:>5.1f}%")
            
            if cumulative_percentage >= 85:
                break
        
        # Add "Other" option for remaining countries
        remaining_percentage = 100 - cumulative_percentage
        print(f"📝 Other countries: {remaining_percentage:.1f}% of companies")
        
        return recommended_countries, country_counts
        
    except Exception as e:
        print(f"❌ Error analyzing countries: {e}")
        return [], {}

def create_enhanced_mode_interface():
    """Create the enhanced mode interface structure."""
    print(f"\n🚀 ENHANCED MODE INTERFACE DESIGN")
    print("=" * 50)
    
    recommended_sectors, sector_counts = analyze_sector_options()
    recommended_countries, country_counts = analyze_country_options()
    
    print(f"\n📋 RECOMMENDED ENHANCED MODE STRUCTURE:")
    print("-" * 50)
    
    # Sector options
    print(f"\n🏢 SECTOR OPTIONS ({len(recommended_sectors)} options):")
    for i, sector in enumerate(recommended_sectors, 1):
        count = sector_counts.get(sector, 0)
        percentage = (count / len(pd.read_csv('202402_Copy.csv'))) * 100
        print(f"  {i:2d}. {sector:<35} ({percentage:>5.1f}%)")
    print(f"  {len(recommended_sectors)+1:2d}. Other")
    
    # Country options
    print(f"\n🌍 COUNTRY OPTIONS ({len(recommended_countries)} options):")
    for i, country in enumerate(recommended_countries, 1):
        count = country_counts.get(country, 0)
        percentage = (count / len(pd.read_csv('202402_Copy.csv'))) * 100
        print(f"  {i:2d}. {country:<25} ({percentage:>5.1f}%)")
    print(f"  {len(recommended_countries)+1:2d}. Other")
    
    # Currency options (simpler)
    print(f"\n💰 CURRENCY OPTIONS (5 options):")
    print("   1. USD (US Dollar)")
    print("   2. EUR (Euro)")
    print("   3. GBP (British Pound)")
    print("   4. CAD (Canadian Dollar)")
    print("   5. Other")
    
    return recommended_sectors, recommended_countries

def generate_api_structure():
    """Generate the API structure for enhanced mode."""
    print(f"\n🔧 API STRUCTURE FOR ENHANCED MODE")
    print("=" * 50)
    
    recommended_sectors, recommended_countries = create_enhanced_mode_interface()
    
    print(f"\n📝 PYDANTIC MODELS:")
    print("-" * 30)
    
    # Enhanced request model
    print("""
class EnhancedGuidedInputRequest(BaseModel):
    # Basic inputs (required)
    company_name: str
    current_arr: float
    net_new_arr: float
    
    # Enhanced mode (optional)
    enhanced_mode: bool = False
    
    # Enhanced inputs (optional, only used if enhanced_mode=True)
    sector: str = None
    country: str = None
    currency: str = None
    
    # Historical data (optional)
    historical_arr: HistoricalARR = None
    
    # Advanced metrics (optional)
    advanced_mode: bool = False
    advanced_metrics: AdvancedMetrics = None
""")
    
    # Validation logic
    print(f"\n🔍 VALIDATION LOGIC:")
    print("-" * 30)
    print("""
# Sector validation
VALID_SECTORS = [
    "Cyber Security",
    "Data & Analytics", 
    "Infrastructure & Network",
    "Communication & Collaboration",
    "Marketing & Customer Experience",
    "Other"
]

# Country validation  
VALID_COUNTRIES = [
    "United States",
    "Israel",
    "Germany", 
    "United Kingdom",
    "France",
    "Other"
]

# Currency validation
VALID_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "Other"]
""")
    
    return recommended_sectors, recommended_countries

def main():
    """Run the complete analysis."""
    print("🚀 CATEGORICAL OPTIONS ANALYSIS FOR ENHANCED MODE")
    print("=" * 70)
    
    # Analyze options
    recommended_sectors, recommended_countries = generate_api_structure()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"\n📋 KEY RECOMMENDATIONS:")
    print(f"  • Sector options: {len(recommended_sectors)} main sectors + 'Other'")
    print(f"  • Country options: {len(recommended_countries)} main countries + 'Other'")
    print(f"  • Currency options: 5 main currencies + 'Other'")
    print(f"  • Enhanced mode: Optional flag to enable detailed inputs")
    print(f"  • Validation: Ensure only valid options are accepted")
    print(f"  • Fallback: Use data-driven defaults if enhanced mode is False")

if __name__ == "__main__":
    main()
