#!/usr/bin/env python3
"""
Enhanced Guided Input System with Historical ARR Support
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedGuidedInputSystem:
    """
    Enhanced guided input system that supports both minimal inputs and historical ARR data.
    """
    
    def __init__(self, training_data_path: str = '202402_Copy.csv'):
        self.training_data_path = training_data_path
        self.relationship_models = {}
        self.is_initialized = False
        
    def initialize_from_training_data(self):
        """Learn relationships from training data."""
        print("🔍 Learning relationships from training data...")
        try:
            df = pd.read_csv(self.training_data_path)
            print(f"✅ Loaded {len(df)} rows of training data")
            
            # Calculate key metrics
            df['Net New ARR'] = df.groupby('id_company')['cARR'].transform(lambda x: x.diff())
            df['ARR_per_Headcount'] = df['cARR'] / df['Headcount (HC)']
            df['Magic_Number'] = df['Net New ARR'] / df.groupby('id_company')['Sales & Marketing'].shift(1)
            df['Burn_Multiple'] = np.abs(df['Cash Burn (OCF & ICF)']) / df['Net New ARR']
            
            # Clean infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Learn relationships
            self._learn_arr_size_relationships(df)
            self._learn_growth_relationships(df)
            
            self.is_initialized = True
            print("✅ Relationships learned successfully!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load training data: {e}")
            self._set_conservative_defaults()
            self.is_initialized = True
    
    def _learn_arr_size_relationships(self, df: pd.DataFrame):
        """Learn relationships based on company size (ARR)."""
        df['ARR_size_category'] = pd.cut(df['cARR'], 
                                       bins=[0, 1e6, 10e6, 100e6, np.inf],
                                       labels=['Small', 'Medium', 'Large', 'Enterprise'])
        
        size_metrics = df.groupby('ARR_size_category').agg({
            'ARR_per_Headcount': 'median',
            'Magic_Number': 'median',
            'Burn_Multiple': 'median',
            'Gross Margin (in %)': 'median',
            'Revenue YoY Growth (in %)': 'median'
        }).fillna(method='ffill')
        
        self.relationship_models['size_relationships'] = size_metrics
    
    def _learn_growth_relationships(self, df: pd.DataFrame):
        """Learn relationships based on growth rates."""
        df['growth_category'] = pd.cut(df['ARR YoY Growth (in %)'], 
                                     bins=[-np.inf, 0, 20, 50, 100, np.inf],
                                     labels=['Declining', 'Slow', 'Moderate', 'Fast', 'Hyper'])
        
        growth_metrics = df.groupby('growth_category').agg({
            'Magic_Number': 'median',
            'Burn_Multiple': 'median',
            'ARR_per_Headcount': 'median'
        }).fillna(method='ffill')
        
        self.relationship_models['growth_relationships'] = growth_metrics
    
    def _set_conservative_defaults(self):
        """Set conservative default relationships."""
        self.relationship_models = {
            'size_relationships': pd.DataFrame({
                'ARR_per_Headcount': [100000, 200000, 250000, 300000],
                'Magic_Number': [0.3, 0.5, 0.7, 0.8],
                'Burn_Multiple': [1.5, 1.2, 1.0, 0.8],
                'Gross Margin (in %)': [65, 75, 80, 85],
                'Revenue YoY Growth (in %)': [15, 25, 35, 45]
            }, index=['Small', 'Medium', 'Large', 'Enterprise']),
            'growth_relationships': pd.DataFrame({
                'Magic_Number': [0.2, 0.4, 0.6, 0.8, 1.0],
                'Burn_Multiple': [2.0, 1.5, 1.2, 1.0, 0.8],
                'ARR_per_Headcount': [80000, 120000, 180000, 250000, 350000]
            }, index=['Declining', 'Slow', 'Moderate', 'Fast', 'Hyper'])
        }
    
    def get_arr_size_category(self, carr: float) -> str:
        """Get ARR size category."""
        if carr < 1e6:
            return 'Small'
        elif carr < 10e6:
            return 'Medium'
        elif carr < 100e6:
            return 'Large'
        else:
            return 'Enterprise'
    
    def get_growth_category(self, growth_rate: float) -> str:
        """Get growth category."""
        if growth_rate < 0:
            return 'Declining'
        elif growth_rate < 20:
            return 'Slow'
        elif growth_rate < 50:
            return 'Moderate'
        elif growth_rate < 100:
            return 'Fast'
        else:
            return 'Hyper'
    
    def create_forecast_input_with_history(self, 
                                         current_arr: float,
                                         net_new_arr: float,
                                         historical_arr: List[float] = None,
                                         advanced_metrics: Dict = None) -> pd.DataFrame:
        """
        Create forecast input with optional historical ARR data.
        
        Args:
            current_arr: Current quarter ARR
            net_new_arr: Current quarter Net New ARR
            historical_arr: List of 4 historical ARR values [Q1, Q2, Q3, Q4] (most recent last)
            advanced_metrics: Optional advanced metrics overrides
        """
        print("🔧 Creating forecast input with historical data...")
        
        # Calculate growth rate
        if historical_arr and len(historical_arr) >= 4:
            # Use actual historical data to calculate growth
            arr_yoy_growth = ((current_arr - historical_arr[0]) / historical_arr[0]) * 100
            print(f"📈 Calculated YoY growth from historical data: {arr_yoy_growth:.1f}%")
        else:
            # Calculate from current quarter only
            arr_yoy_growth = (net_new_arr / (current_arr - net_new_arr)) * 100 if (current_arr - net_new_arr) > 0 else 0
            print(f"📈 Calculated YoY growth from current quarter: {arr_yoy_growth:.1f}%")
        
        # Infer secondary metrics
        inferred_metrics = self._infer_secondary_metrics(current_arr, net_new_arr, arr_yoy_growth)
        
        # Apply advanced overrides
        if advanced_metrics:
            for key, value in advanced_metrics.items():
                if value is not None and key in inferred_metrics:
                    inferred_metrics[key] = value
                    print(f"🔧 Advanced override: {key} = {value}")
        
        # Create historical quarters
        quarters = []
        
        if historical_arr and len(historical_arr) >= 4:
            # Use provided historical data
            print("📊 Using provided historical ARR data...")
            historical_quarters = [
                ('FY23 Q1', historical_arr[0]),
                ('FY23 Q2', historical_arr[1]),
                ('FY23 Q3', historical_arr[2]),
                ('FY23 Q4', historical_arr[3])
            ]
            
            # Add current quarter
            current_quarter = ('FY24 Q4', current_arr)
            
            # Create quarters with actual historical data
            for i, (quarter_name, arr_value) in enumerate(historical_quarters + [current_quarter]):
                quarter_data = inferred_metrics.copy()
                quarter_data['Financial Quarter'] = quarter_name
                quarter_data['Quarter Num'] = (i % 4) + 1
                quarter_data['cARR'] = arr_value
                
                # Calculate Net New ARR for historical quarters
                if i == 0:
                    quarter_data['Net New ARR'] = 0  # First quarter
                else:
                    prev_arr = historical_quarters[i-1][1] if i <= 4 else historical_quarters[3][1]
                    quarter_data['Net New ARR'] = arr_value - prev_arr
                
                # Scale other metrics proportionally
                scale_factor = arr_value / current_arr
                quarter_data = self._scale_metrics(quarter_data, scale_factor)
                
                quarters.append(quarter_data)
        else:
            # Generate synthetic historical data
            print("📊 Generating synthetic historical data...")
            quarters = self._generate_synthetic_history(current_arr, arr_yoy_growth, inferred_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(quarters)
        
        # Ensure all required columns
        required_columns = [
            'id_company', 'Financial Quarter', 'cARR', 'ARR YoY Growth (in %)',
            'Revenue YoY Growth (in %)', 'Gross Margin (in %)', 'EBITDA',
            'Cash Burn (OCF & ICF)', 'LTM Rule of 40% (ARR)', 'Sales & Marketing',
            'Headcount (HC)', 'Customers (EoP)', 'Expansion & Upsell',
            'Churn & Reduction', 'Quarter Num', 'Net New ARR'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    def _infer_secondary_metrics(self, carr: float, net_new_arr: float, arr_yoy_growth: float) -> Dict:
        """Infer secondary metrics using learned relationships."""
        if not self.is_initialized:
            self.initialize_from_training_data()
        
        # Get relationships
        size_rels = self.relationship_models['size_relationships']
        growth_rels = self.relationship_models['growth_relationships']
        
        # Determine categories
        size_category = self.get_arr_size_category(carr)
        growth_category = self.get_growth_category(arr_yoy_growth)
        
        # Get learned relationships
        try:
            arr_per_headcount = size_rels.loc[size_category, 'ARR_per_Headcount']
            magic_number = size_rels.loc[size_category, 'Magic_Number']
            burn_multiple = size_rels.loc[size_category, 'Burn_Multiple']
            gross_margin = size_rels.loc[size_category, 'Gross Margin (in %)']
            
            # Adjust based on growth rate
            growth_magic_number = growth_rels.loc[growth_category, 'Magic_Number']
            growth_burn_multiple = growth_rels.loc[growth_category, 'Burn_Multiple']
            
            # Use weighted averages
            magic_number = (magic_number + growth_magic_number) / 2
            burn_multiple = (burn_multiple + growth_burn_multiple) / 2
            
        except Exception:
            # Fallback to conservative defaults
            if carr < 1e6:
                arr_per_headcount = 150000
                magic_number = 0.3
                burn_multiple = 1.5
                gross_margin = 65
            elif carr < 10e6:
                arr_per_headcount = 200000
                magic_number = 0.5
                burn_multiple = 1.2
                gross_margin = 75
            else:
                arr_per_headcount = 250000
                magic_number = 0.7
                burn_multiple = 1.0
                gross_margin = 80
        
        # Calculate metrics
        headcount = max(1, min(500, int(carr / arr_per_headcount)))
        sales_marketing = net_new_arr / magic_number if magic_number > 0 else carr * 0.4
        cash_burn = -net_new_arr * burn_multiple if burn_multiple > 0 else -carr * 0.3
        
        return {
            'cARR': carr,
            'Net New ARR': net_new_arr,
            'ARR YoY Growth (in %)': arr_yoy_growth,
            'Sales & Marketing': sales_marketing,
            'EBITDA': carr * 0.2,
            'Cash Burn (OCF & ICF)': cash_burn,
            'LTM Rule of 40% (ARR)': arr_yoy_growth + gross_margin * 0.2,
            'Revenue YoY Growth (in %)': arr_yoy_growth * 0.8,
            'Gross Margin (in %)': gross_margin,
            'Headcount (HC)': headcount,
            'Customers (EoP)': headcount * 5,
            'Expansion & Upsell': net_new_arr * 0.3,
            'Churn & Reduction': -net_new_arr * 0.1,
            'Magic_Number': magic_number,
            'Burn_Multiple': burn_multiple,
            'Net Profit/Loss Margin (in %)': -10,
            'id_company': 'User Company',
            'Quarter Num': 4
        }
    
    def _scale_metrics(self, metrics: Dict, scale_factor: float) -> Dict:
        """Scale metrics proportionally based on ARR."""
        scaled_metrics = metrics.copy()
        
        # Scale ARR-dependent metrics
        arr_dependent = ['Sales & Marketing', 'EBITDA', 'Cash Burn (OCF & ICF)', 
                        'Expansion & Upsell', 'Churn & Reduction']
        
        for metric in arr_dependent:
            if metric in scaled_metrics:
                scaled_metrics[metric] *= scale_factor
        
        # Scale headcount with square root (more realistic)
        if 'Headcount (HC)' in scaled_metrics:
            scaled_metrics['Headcount (HC)'] = max(1, int(scaled_metrics['Headcount (HC)'] * (scale_factor ** 0.5)))
        
        return scaled_metrics
    
    def _generate_synthetic_history(self, current_arr: float, growth_rate: float, base_metrics: Dict) -> List[Dict]:
        """Generate synthetic historical data."""
        quarters = []
        growth_decimal = growth_rate / 100
        
        # Generate 4 historical quarters
        for i in range(4):
            quarter_num = i + 1
            year = 23 if quarter_num <= 4 else 24
            quarter_name = f'FY{year} Q{quarter_num}'
            
            # Calculate historical ARR
            quarters_back = 4 - i
            historical_arr = current_arr / ((1 + growth_decimal) ** quarters_back)
            
            # Create quarter data
            quarter_data = base_metrics.copy()
            quarter_data['Financial Quarter'] = quarter_name
            quarter_data['Quarter Num'] = quarter_num
            quarter_data['cARR'] = historical_arr
            quarter_data['Net New ARR'] = historical_arr * growth_decimal if quarter_num > 1 else 0
            
            # Scale other metrics
            scale_factor = historical_arr / current_arr
            quarter_data = self._scale_metrics(quarter_data, scale_factor)
            
            quarters.append(quarter_data)
        
        # Add current quarter
        current_quarter = base_metrics.copy()
        current_quarter['Financial Quarter'] = 'FY24 Q4'
        current_quarter['Quarter Num'] = 4
        quarters.append(current_quarter)
        
        return quarters

# Example usage
if __name__ == "__main__":
    # Initialize system
    enhanced_system = EnhancedGuidedInputSystem()
    
    # Test with minimal inputs
    print("🧪 Testing with minimal inputs...")
    df_minimal = enhanced_system.create_forecast_input_with_history(
        current_arr=2100000,
        net_new_arr=320000
    )
    print(f"✅ Created {len(df_minimal)} quarters with minimal inputs")
    
    # Test with historical data
    print("\n🧪 Testing with historical ARR data...")
    historical_arr = [1500000, 1700000, 1900000, 2000000]  # Q1, Q2, Q3, Q4
    df_historical = enhanced_system.create_forecast_input_with_history(
        current_arr=2100000,
        net_new_arr=320000,
        historical_arr=historical_arr
    )
    print(f"✅ Created {len(df_historical)} quarters with historical data")
    
    # Test with advanced metrics
    print("\n🧪 Testing with advanced metrics...")
    advanced_metrics = {
        'magic_number': 0.95,
        'gross_margin': 82.0,
        'headcount': 38
    }
    df_advanced = enhanced_system.create_forecast_input_with_history(
        current_arr=2100000,
        net_new_arr=320000,
        historical_arr=historical_arr,
        advanced_metrics=advanced_metrics
    )
    print(f"✅ Created {len(df_advanced)} quarters with advanced metrics")
