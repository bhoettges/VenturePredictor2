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
            # Use actual historical data to calculate YoY growth
            arr_yoy_growth = ((current_arr - historical_arr[0]) / historical_arr[0]) * 100
            print(f"📈 Calculated YoY growth from historical data: {arr_yoy_growth:.1f}%")
        else:
            # Calculate quarterly growth rate and convert to YoY equivalent
            quarterly_growth = net_new_arr / (current_arr - net_new_arr) if (current_arr - net_new_arr) > 0 else 0
            # Convert quarterly growth to YoY growth: (1 + q)^4 - 1
            yoy_growth = ((1 + quarterly_growth) ** 4 - 1) * 100
            arr_yoy_growth = yoy_growth
            print(f"📈 Quarterly growth: {quarterly_growth*100:.1f}% → YoY equivalent: {yoy_growth:.1f}%")
        
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
            # Generate realistic historical data that shows the growth pattern
            print("📊 Generating realistic historical data with growth pattern...")
            quarters = self._generate_realistic_history(current_arr, net_new_arr, inferred_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(quarters)
        
        # Ensure all required columns from the original training data
        all_training_columns = [
            'id', 'id_company', 'Currency', 'id_currency', 'Sector', 'id_sector', 
            'Target Customer', 'id_target_customer', 'Country', 'id_country', 
            'Deal Team', 'id_deal_team', 'Financial Quarter', 'End of Quarter', 
            'Revenue Run Rate (RRR)', 'Net New ARR', 'cARR', 'ARR YoY Growth (in %)', 
            'Revenue', 'LTM Revenue', 'Revenue as % of ARR', 'Revenue YoY Growth (in %)', 
            'Subscription', 'Professional Services', 'Other Non-Recurring', 'COGS', 
            'Gross Profit', 'Gross Margin (in %)', 'Operating Expenses', 
            'Opex as % of Revenue', 'Opex YoY Growth (in %)', 'Sales & Marketing', 
            'S&M as % of Revenue', 'S&M Expenses YoY Growth (in %)', 
            'Research & Development', 'R&D as % of Revenue', 
            'R&D Expenses YoY Growth (in %)', 'General & Administrative', 
            'G&A as % of Revenue', 'G&A Expenses YoY Growth (in %)', 'EBITDA', 
            'Depreciation and Amortization', 'EBIT', 'Interest Income', 
            'Interest Expenses', 'Taxes', 'Other', 'Net Income', 
            'Cash Burn (OCF & ICF)', 'Monthly Burn (Qtly. avg)', 
            'Financing Cash Flow (FCF)', 'Ending cash', 'Cash Runway (Months)', 
            'Total Current Assets (excluding Cash)', 'Total Current Liabilities', 
            'Working Capital', 'Financial Debt', 'New ARR', 'Expansion & Upsell', 
            'Churn & Reduction', 'Gross New ARR', 'GNARR YoY Growth (in %)', 
            'Expansion & Upsell (in %)', 'Churn & Reduction (in %)', 
            'Annualized $ Expansion & Upsell (in %)', 
            'Annualized $ Churn & Reduction (in %)', 
            'Expansion & Upsell % of Gross New ARR', 
            'Annualized $ Net Expansion (in %)', 'Net Expansion TTM (in %)', 
            'Quick Ratio', 'Quick Ratio TTM', 'LTM Capital Efficiency (ARR)', 
            'LTM New Revenue/ Cash Burn', 'LTM GP / LTM Cash Burn', 
            'LTM GP / LTM S&M', 'Capital Efficiency (NNARR/Cash Burn)', 
            'LTM Rule of 40% (ARR)', 'LTM Rule of 40% (Rev)', 
            'LTM Magic Number (ARR)', 'LTM Magic Number (Rev)', 
            'ACV (EoP, Aggregate)', 'ACV YoY Growth (in %)', 
            'New ARR / New Customers', 'New ARR / New Customers YoY Growth (in %)', 
            'CAC Payback (months)', 'LTV/CAC (TTM)', 'LTV/CAC', 'CAC', 'LTV', 
            'Customer Lifetime', 'New Customers', 'Churned Customers', 
            'Customers (EoP)', 'Annualized Logo Churn (in %)', 'Headcount (HC)', 
            'HC YoY Growth (in %)', 'Sales Reps', 'Sales Reps as % of HC', 
            'Sales Reps YoY Growth (in %)', 'Research & Development HC', 
            'R&D HC as % of HC', 'R&D HC YoY Growth (in %)', 
            'General & Administrative HC', 'G&A as % of HC', 
            'G&A YoY Growth (in %)', 'ARR / HC', 'ARR / HC.1', 'RRR / HC', 
            'Opex Run Rate / HC', 'Cash Burn / HC', 'NNARR / Sales Reps', 
            'EqV', 'TEV/ARR', 'TEV/ARR Growth Adj.', 'TEV/ARR (end of cash)', 
            'TEV/LTM Revenue', 'TEV/LTM Revenue Growth Adj.', 
            'TEV/LTM Revenue (end of cash)', 'Quarter Order', 'Quarter_old', 
            'Date', 'Order', 'Quarter Num'
        ]
        
        # Add missing columns with appropriate defaults
        for col in all_training_columns:
            if col not in df.columns:
                df[col] = self._get_default_value(col, inferred_metrics)
        
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
    
    def _generate_synthetic_history(self, current_arr: float, yoy_growth_rate: float, base_metrics: Dict) -> List[Dict]:
        """Generate synthetic historical data."""
        quarters = []
        
        # Convert YoY growth to quarterly growth for historical generation
        quarterly_growth = ((1 + yoy_growth_rate/100) ** (1/4) - 1)
        
        # Generate 4 historical quarters
        for i in range(4):
            quarter_num = i + 1
            year = 23 if quarter_num <= 4 else 24
            quarter_name = f'FY{year} Q{quarter_num}'
            
            # Calculate historical ARR using quarterly growth
            quarters_back = 4 - i
            historical_arr = current_arr / ((1 + quarterly_growth) ** quarters_back)
            
            # Create quarter data
            quarter_data = base_metrics.copy()
            quarter_data['Financial Quarter'] = quarter_name
            quarter_data['Quarter Num'] = quarter_num
            quarter_data['cARR'] = historical_arr
            quarter_data['Net New ARR'] = historical_arr * quarterly_growth if quarter_num > 1 else 0
            
            # For YoY growth, calculate the actual YoY growth for each quarter
            if quarter_num == 1:
                # Q1: compare to Q1 of previous year (synthetic)
                prev_year_q1 = historical_arr / ((1 + quarterly_growth) ** 4)
                quarter_data['ARR YoY Growth (in %)'] = ((historical_arr - prev_year_q1) / prev_year_q1) * 100
            else:
                # For other quarters, use a scaled version of the YoY growth
                quarter_data['ARR YoY Growth (in %)'] = yoy_growth_rate * (0.8 + 0.2 * (quarter_num / 4))
            
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
    
    def _generate_realistic_history(self, current_arr: float, net_new_arr: float, base_metrics: Dict) -> List[Dict]:
        """Generate realistic historical data that shows the actual growth pattern."""
        quarters = []
        
        # Calculate the quarterly growth rate from current data
        quarterly_growth = net_new_arr / (current_arr - net_new_arr) if (current_arr - net_new_arr) > 0 else 0
        print(f"📈 Detected quarterly growth rate: {quarterly_growth*100:.1f}%")
        
        # Generate 4 historical quarters showing this growth pattern
        for i in range(4):
            quarter_num = i + 1
            year = 23 if quarter_num <= 4 else 24
            quarter_name = f'FY{year} Q{quarter_num}'
            
            # Calculate historical ARR using the detected growth rate
            quarters_back = 4 - i
            historical_arr = current_arr / ((1 + quarterly_growth) ** quarters_back)
            
            # Create quarter data
            quarter_data = base_metrics.copy()
            quarter_data['Financial Quarter'] = quarter_name
            quarter_data['Quarter Num'] = quarter_num
            quarter_data['cARR'] = historical_arr
            
            # Calculate Net New ARR for this quarter
            if quarter_num == 1:
                quarter_data['Net New ARR'] = 0  # First quarter
            else:
                prev_arr = current_arr / ((1 + quarterly_growth) ** (quarters_back + 1))
                quarter_data['Net New ARR'] = historical_arr - prev_arr
            
            # Scale other metrics proportionally
            scale_factor = historical_arr / current_arr
            quarter_data = self._scale_metrics(quarter_data, scale_factor)
            
            quarters.append(quarter_data)
        
        # Add current quarter
        current_quarter = base_metrics.copy()
        current_quarter['Financial Quarter'] = 'FY24 Q4'
        current_quarter['Quarter Num'] = 4
        quarters.append(current_quarter)
        
        return quarters
    
    def _get_default_value(self, column_name: str, inferred_metrics: Dict) -> float:
        """Get intelligent default values for missing columns based on the column name and inferred metrics."""
        
        # Get base metrics
        carr = inferred_metrics.get('cARR', 1000000)
        net_new_arr = inferred_metrics.get('Net New ARR', 100000)
        arr_yoy_growth = inferred_metrics.get('ARR YoY Growth (in %)', 20)
        headcount = inferred_metrics.get('Headcount (HC)', 50)
        sales_marketing = inferred_metrics.get('Sales & Marketing', 200000)
        gross_margin = inferred_metrics.get('Gross Margin (in %)', 75)
        
        # ID and categorical features
        if column_name in ['id', 'id_company']:
            return 99999  # Unique ID for user company
        elif column_name in ['Currency', 'id_currency']:
            return 1  # USD
        elif column_name in ['Sector', 'id_sector']:
            return 1  # Technology
        elif column_name in ['Target Customer', 'id_target_customer']:
            return 1  # Enterprise
        elif column_name in ['Country', 'id_country']:
            return 1  # United States
        elif column_name in ['Deal Team', 'id_deal_team']:
            return 1  # Default team
        
        # Time-based features
        elif column_name in ['Financial Quarter', 'End of Quarter']:
            return 'FY24 Q4'
        elif column_name in ['Quarter Order', 'Quarter_old', 'Date', 'Order']:
            return 1
        
        # Revenue and ARR related
        elif column_name == 'Revenue Run Rate (RRR)':
            return carr * 1.1  # RRR slightly higher than ARR
        elif column_name == 'Revenue':
            return carr * 0.9  # Revenue typically 90% of ARR
        elif column_name == 'LTM Revenue':
            return carr * 3.6  # LTM = 4 quarters * 0.9
        elif column_name == 'Revenue as % of ARR':
            return 90.0
        elif column_name == 'Subscription':
            return carr * 0.8  # 80% subscription revenue
        elif column_name == 'Professional Services':
            return carr * 0.1  # 10% professional services
        elif column_name == 'Other Non-Recurring':
            return carr * 0.05  # 5% other revenue
        
        # Cost and margin related
        elif column_name == 'COGS':
            return carr * (1 - gross_margin/100)  # COGS = Revenue * (1 - Gross Margin)
        elif column_name == 'Gross Profit':
            return carr * (gross_margin/100)  # Gross Profit = Revenue * Gross Margin
        elif column_name == 'Operating Expenses':
            return carr * 0.6  # 60% of revenue
        elif column_name == 'Opex as % of Revenue':
            return 60.0
        elif column_name == 'Opex YoY Growth (in %)':
            return arr_yoy_growth * 0.8  # Opex grows slower than revenue
        
        # Sales & Marketing
        elif column_name == 'S&M as % of Revenue':
            return (sales_marketing / carr) * 100
        elif column_name == 'S&M Expenses YoY Growth (in %)':
            return arr_yoy_growth * 0.9
        
        # R&D
        elif column_name == 'Research & Development':
            return carr * 0.2  # 20% of revenue
        elif column_name == 'R&D as % of Revenue':
            return 20.0
        elif column_name == 'R&D Expenses YoY Growth (in %)':
            return arr_yoy_growth * 0.8
        elif column_name == 'Research & Development HC':
            return int(headcount * 0.3)  # 30% of headcount
        elif column_name == 'R&D HC as % of HC':
            return 30.0
        elif column_name == 'R&D HC YoY Growth (in %)':
            return arr_yoy_growth * 0.7
        
        # G&A
        elif column_name == 'General & Administrative':
            return carr * 0.15  # 15% of revenue
        elif column_name == 'G&A as % of Revenue':
            return 15.0
        elif column_name == 'G&A Expenses YoY Growth (in %)':
            return arr_yoy_growth * 0.6
        elif column_name == 'General & Administrative HC':
            return int(headcount * 0.2)  # 20% of headcount
        elif column_name == 'G&A as % of HC':
            return 20.0
        elif column_name == 'G&A YoY Growth (in %)':
            return arr_yoy_growth * 0.5
        
        # Profitability
        elif column_name == 'EBITDA':
            return carr * 0.2  # 20% EBITDA margin
        elif column_name == 'Depreciation and Amortization':
            return carr * 0.05  # 5% of revenue
        elif column_name == 'EBIT':
            return carr * 0.15  # 15% EBIT margin
        elif column_name in ['Interest Income', 'Interest Expenses']:
            return carr * 0.01  # 1% of revenue
        elif column_name == 'Taxes':
            return carr * 0.03  # 3% tax rate
        elif column_name == 'Other':
            return carr * 0.02  # 2% other income/expenses
        elif column_name == 'Net Income':
            return carr * 0.12  # 12% net margin
        
        # Cash flow
        elif column_name == 'Monthly Burn (Qtly. avg)':
            return abs(inferred_metrics.get('Cash Burn (OCF & ICF)', -100000)) / 3
        elif column_name == 'Financing Cash Flow (FCF)':
            return carr * 0.1  # 10% of revenue
        elif column_name == 'Ending cash':
            return carr * 0.5  # 6 months of revenue
        elif column_name == 'Cash Runway (Months)':
            return 18.0  # 18 months runway
        
        # Balance sheet
        elif column_name == 'Total Current Assets (excluding Cash)':
            return carr * 0.3  # 30% of revenue
        elif column_name == 'Total Current Liabilities':
            return carr * 0.2  # 20% of revenue
        elif column_name == 'Working Capital':
            return carr * 0.1  # 10% of revenue
        elif column_name == 'Financial Debt':
            return carr * 0.1  # 10% of revenue
        
        # ARR metrics
        elif column_name == 'New ARR':
            return net_new_arr
        elif column_name == 'Gross New ARR':
            return net_new_arr * 1.2  # 20% higher than net new
        elif column_name == 'GNARR YoY Growth (in %)':
            return arr_yoy_growth * 1.1
        
        # Customer metrics
        elif column_name == 'New Customers':
            return int(headcount * 0.1)  # 10% of headcount
        elif column_name == 'Churned Customers':
            return int(headcount * 0.02)  # 2% churn
        elif column_name == 'New ARR / New Customers':
            return net_new_arr / max(1, int(headcount * 0.1))
        elif column_name == 'New ARR / New Customers YoY Growth (in %)':
            return arr_yoy_growth * 0.9
        elif column_name == 'Annualized Logo Churn (in %)':
            return 5.0  # 5% annual churn
        
        # Headcount metrics
        elif column_name == 'HC YoY Growth (in %)':
            return arr_yoy_growth * 0.6  # Headcount grows slower than revenue
        elif column_name == 'Sales Reps':
            return int(headcount * 0.2)  # 20% sales reps
        elif column_name == 'Sales Reps as % of HC':
            return 20.0
        elif column_name == 'Sales Reps YoY Growth (in %)':
            return arr_yoy_growth * 0.8
        
        # Efficiency ratios
        elif column_name in ['ARR / HC', 'ARR / HC.1']:
            return carr / max(1, headcount)
        elif column_name == 'RRR / HC':
            return (carr * 1.1) / max(1, headcount)
        elif column_name == 'Opex Run Rate / HC':
            return (carr * 0.6) / max(1, headcount)
        elif column_name == 'Cash Burn / HC':
            return abs(inferred_metrics.get('Cash Burn (OCF & ICF)', -100000)) / max(1, headcount)
        elif column_name == 'NNARR / Sales Reps':
            return net_new_arr / max(1, int(headcount * 0.2))
        
        # Valuation metrics
        elif column_name == 'EqV':
            return carr * 8  # 8x ARR valuation
        elif column_name == 'TEV/ARR':
            return 8.0
        elif column_name == 'TEV/ARR Growth Adj.':
            return 7.0
        elif column_name == 'TEV/ARR (end of cash)':
            return 6.0
        elif column_name == 'TEV/LTM Revenue':
            return 7.0
        elif column_name == 'TEV/LTM Revenue Growth Adj.':
            return 6.0
        elif column_name == 'TEV/LTM Revenue (end of cash)':
            return 5.0
        
        # Customer metrics
        elif column_name == 'ACV (EoP, Aggregate)':
            return carr * 0.8  # 80% of ARR
        elif column_name == 'ACV YoY Growth (in %)':
            return arr_yoy_growth * 0.9
        elif column_name == 'CAC':
            return sales_marketing / max(1, int(headcount * 0.1))
        elif column_name == 'LTV':
            return (carr * 0.8) * 3  # 3x ACV LTV
        elif column_name == 'Customer Lifetime':
            return 36.0  # 36 months
        elif column_name == 'CAC Payback (months)':
            return 12.0  # 12 months
        elif column_name in ['LTV/CAC', 'LTV/CAC (TTM)']:
            return 3.0  # 3x LTV/CAC
        
        # Expansion and churn metrics
        elif column_name == 'Expansion & Upsell (in %)':
            return 20.0  # 20% expansion
        elif column_name == 'Churn & Reduction (in %)':
            return -5.0  # -5% churn
        elif column_name == 'Annualized $ Expansion & Upsell (in %)':
            return 20.0
        elif column_name == 'Annualized $ Churn & Reduction (in %)':
            return -5.0
        elif column_name == 'Expansion & Upsell % of Gross New ARR':
            return 25.0
        elif column_name == 'Annualized $ Net Expansion (in %)':
            return 15.0
        elif column_name == 'Net Expansion TTM (in %)':
            return 15.0
        
        # Financial ratios
        elif column_name in ['Quick Ratio', 'Quick Ratio TTM']:
            return 2.0  # 2x quick ratio
        elif column_name == 'LTM Capital Efficiency (ARR)':
            return 0.8  # 80% efficiency
        elif column_name == 'LTM New Revenue/ Cash Burn':
            return 1.2  # 1.2x efficiency
        elif column_name == 'LTM GP / LTM Cash Burn':
            return 1.5  # 1.5x efficiency
        elif column_name == 'LTM GP / LTM S&M':
            return 2.0  # 2x efficiency
        elif column_name == 'Capital Efficiency (NNARR/Cash Burn)':
            return 0.8  # 80% efficiency
        elif column_name == 'LTM Rule of 40% (Rev)':
            return 35.0  # 35% rule of 40
        elif column_name == 'LTM Magic Number (ARR)':
            return 0.8  # 0.8 magic number
        elif column_name == 'LTM Magic Number (Rev)':
            return 0.7  # 0.7 magic number
        
        # Default fallback
        else:
            return 0.0

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
