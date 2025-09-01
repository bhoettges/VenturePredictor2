import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class GuidedInputSystem:
    """
    Advanced guided input system with intelligent defaults for financial forecasting.
    Asks users for only critical inputs and intelligently infers secondary metrics.
    """
    
    def __init__(self, training_data_path: str = '202402_Copy.csv'):
        """
        Initialize the guided input system.
        
        Args:
            training_data_path: Path to the training dataset for learning relationships
        """
        self.training_data_path = training_data_path
        self.relationship_models = {}
        self.scaler = StandardScaler()
        self.is_initialized = False
        
    def initialize_from_training_data(self):
        """Learn relationships from training data to enable intelligent defaults."""
        print("🔍 Learning relationships from training data...")
        print(f"📁 Looking for training data at: {self.training_data_path}")
        
        try:
            # Load training data
            df = pd.read_csv(self.training_data_path)
            print(f"✅ Loaded {len(df)} rows of training data")
            print(f"📊 Found {df['id_company'].nunique()} unique companies")
            
            # Calculate key metrics
            df['Net New ARR'] = df.groupby('id_company')['cARR'].transform(lambda x: x.diff())
            df['ARR_per_Headcount'] = df['cARR'] / df['Headcount (HC)']
            df['Magic_Number'] = df['Net New ARR'] / df.groupby('id_company')['Sales & Marketing'].shift(1)
            df['Burn_Multiple'] = np.abs(df['Cash Burn (OCF & ICF)']) / df['Net New ARR']
            
            # Clean infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Learn relationships for intelligent defaults
            self._learn_arr_size_relationships(df)
            self._learn_growth_relationships(df)
            self._learn_efficiency_relationships(df)
            
            # Show what we learned
            print("📈 Learned relationships:")
            print(f"  Size categories: {list(self.relationship_models['size_relationships'].index)}")
            print(f"  Growth categories: {list(self.relationship_models['growth_relationships'].index)}")
            
            self.is_initialized = True
            print("✅ Relationships learned successfully from your data!")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load training data: {e}")
            print("Using conservative default relationships...")
            self._set_conservative_defaults()
            self.is_initialized = True
    
    def _learn_arr_size_relationships(self, df: pd.DataFrame):
        """Learn relationships based on company size (ARR)."""
        # Group companies by ARR size
        df['ARR_size_category'] = pd.cut(df['cARR'], 
                                       bins=[0, 1e6, 10e6, 100e6, np.inf],
                                       labels=['Small', 'Medium', 'Large', 'Enterprise'])
        
        # Calculate averages by size category
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
        # Group by growth rate
        df['growth_category'] = pd.cut(df['ARR YoY Growth (in %)'], 
                                     bins=[-np.inf, 0, 20, 50, 100, np.inf],
                                     labels=['Declining', 'Slow', 'Moderate', 'Fast', 'Hyper'])
        
        growth_metrics = df.groupby('growth_category').agg({
            'Magic_Number': 'median',
            'Burn_Multiple': 'median',
            'ARR_per_Headcount': 'median'
        }).fillna(method='ffill')
        
        self.relationship_models['growth_relationships'] = growth_metrics
    
    def _learn_efficiency_relationships(self, df: pd.DataFrame):
        """Learn relationships based on efficiency metrics."""
        # Magic Number vs other metrics
        df['efficiency_category'] = pd.cut(df['Magic_Number'], 
                                         bins=[-np.inf, 0, 0.5, 1, 2, np.inf],
                                         labels=['Poor', 'Low', 'Moderate', 'Good', 'Excellent'])
        
        efficiency_metrics = df.groupby('efficiency_category').agg({
            'Burn_Multiple': 'median',
            'ARR_per_Headcount': 'median',
            'Gross Margin (in %)': 'median'
        }).fillna(method='ffill')
        
        self.relationship_models['efficiency_relationships'] = efficiency_metrics
    
    def _set_conservative_defaults(self):
        """Set conservative default relationships when training data is unavailable."""
        self.relationship_models = {
            'size_relationships': pd.DataFrame({
                'ARR_per_Headcount': [150000, 200000, 250000, 300000],
                'Magic_Number': [0.3, 0.5, 0.7, 0.9],
                'Burn_Multiple': [2.0, 1.5, 1.0, 0.8],
                'Gross Margin (in %)': [70, 75, 80, 85],
                'Revenue YoY Growth (in %)': [20, 30, 40, 50]
            }, index=['Small', 'Medium', 'Large', 'Enterprise']),
            
            'growth_relationships': pd.DataFrame({
                'Magic_Number': [0.2, 0.4, 0.6, 0.8, 1.0],
                'Burn_Multiple': [2.5, 2.0, 1.5, 1.0, 0.8],
                'ARR_per_Headcount': [120000, 150000, 200000, 250000, 300000]
            }, index=['Declining', 'Slow', 'Moderate', 'Fast', 'Hyper']),
            
            'efficiency_relationships': pd.DataFrame({
                'Burn_Multiple': [3.0, 2.0, 1.5, 1.0, 0.8],
                'ARR_per_Headcount': [100000, 150000, 200000, 250000, 300000],
                'Gross Margin (in %)': [60, 70, 75, 80, 85]
            }, index=['Poor', 'Low', 'Moderate', 'Good', 'Excellent'])
        }
    
    def get_arr_size_category(self, carr: float) -> str:
        """Determine ARR size category."""
        if carr < 1e6:
            return 'Small'
        elif carr < 10e6:
            return 'Medium'
        elif carr < 100e6:
            return 'Large'
        else:
            return 'Enterprise'
    
    def get_growth_category(self, growth_rate: float) -> str:
        """Determine growth category."""
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
    
    def infer_secondary_metrics(self, primary_inputs: Dict) -> Dict:
        """
        Infer secondary metrics based on primary inputs and learned relationships.
        
        Args:
            primary_inputs: Dictionary with primary inputs (cARR, Net New ARR, etc.)
            
        Returns:
            Dictionary with all metrics (primary + inferred)
        """
        if not self.is_initialized:
            self.initialize_from_training_data()
        
        # Extract primary inputs
        carr = primary_inputs.get('cARR', 0)
        net_new_arr = primary_inputs.get('Net New ARR', 0)
        arr_yoy_growth = primary_inputs.get('ARR YoY Growth (in %)', 0)
        
        # Determine categories
        size_category = self.get_arr_size_category(carr)
        growth_category = self.get_growth_category(arr_yoy_growth)
        
        # Get relationship data
        size_rels = self.relationship_models['size_relationships']
        growth_rels = self.relationship_models['growth_relationships']
        efficiency_rels = self.relationship_models['efficiency_relationships']
        
        # Infer metrics based on size
        arr_per_headcount = size_rels.loc[size_category, 'ARR_per_Headcount']
        magic_number = size_rels.loc[size_category, 'Magic_Number']
        burn_multiple = size_rels.loc[size_category, 'Burn_Multiple']
        gross_margin = size_rels.loc[size_category, 'Gross Margin (in %)']
        revenue_growth = size_rels.loc[size_category, 'Revenue YoY Growth (in %)']
        
        # Adjust based on growth rate
        growth_magic_number = growth_rels.loc[growth_category, 'Magic_Number']
        growth_burn_multiple = growth_rels.loc[growth_category, 'Burn_Multiple']
        
        # Use weighted average of size and growth relationships
        magic_number = (magic_number + growth_magic_number) / 2
        burn_multiple = (burn_multiple + growth_burn_multiple) / 2
        
        # Calculate inferred metrics using learned relationships from your data
        # Get relationships learned from training data
        size_rels = self.relationship_models['size_relationships']
        growth_rels = self.relationship_models['growth_relationships']
        
        # Determine categories
        size_category = self.get_arr_size_category(carr)
        growth_category = self.get_growth_category(arr_yoy_growth)
        
        print(f"🔍 Using learned relationships for {size_category} company (ARR: ${carr:,.0f})")
        print(f"🔍 Growth category: {growth_category} (Growth: {arr_yoy_growth:.1f}%)")
        
        # Get learned relationships from your data
        try:
            arr_per_headcount = size_rels.loc[size_category, 'ARR_per_Headcount']
            magic_number = size_rels.loc[size_category, 'Magic_Number']
            burn_multiple = size_rels.loc[size_category, 'Burn_Multiple']
            gross_margin = size_rels.loc[size_category, 'Gross Margin (in %)']
            
            # Adjust based on growth rate from your data
            growth_magic_number = growth_rels.loc[growth_category, 'Magic_Number']
            growth_burn_multiple = growth_rels.loc[growth_category, 'Burn_Multiple']
            
            # Use weighted average of size and growth relationships
            magic_number = (magic_number + growth_magic_number) / 2
            burn_multiple = (burn_multiple + growth_burn_multiple) / 2
            
            print(f"📊 Learned from your data:")
            print(f"  ARR per Headcount: ${arr_per_headcount:,.0f}")
            print(f"  Magic Number: {magic_number:.2f}")
            print(f"  Gross Margin: {gross_margin:.1f}%")
            
            # Validate learned values and use industry benchmarks if unrealistic
            if arr_per_headcount < 10000 or arr_per_headcount > 1000000:
                print(f"⚠️ ARR per headcount (${arr_per_headcount:,.0f}) seems unrealistic, using industry benchmark")
                if carr < 1000000:
                    arr_per_headcount = 150000
                elif carr < 10000000:
                    arr_per_headcount = 200000
                else:
                    arr_per_headcount = 250000
            
            if gross_margin < 20 or gross_margin > 95:
                print(f"⚠️ Gross margin ({gross_margin:.1f}%) seems unrealistic, using industry benchmark")
                if carr < 1000000:
                    gross_margin = 65
                elif carr < 10000000:
                    gross_margin = 75
                else:
                    gross_margin = 80
            
            if magic_number < 0 or magic_number > 5:
                print(f"⚠️ Magic number ({magic_number:.2f}) seems unrealistic, using industry benchmark")
                if carr < 1000000:
                    magic_number = 0.3
                elif carr < 10000000:
                    magic_number = 0.5
                else:
                    magic_number = 0.7
            
        except Exception as e:
            print(f"⚠️ Could not use learned relationships: {e}")
            print("🔍 Falling back to industry benchmarks...")
            # Fallback to industry benchmarks
            if carr < 100000:
                arr_per_headcount = 50000
                magic_number = 0.2
                gross_margin = 60
            elif carr < 1000000:
                arr_per_headcount = 100000
                magic_number = 0.3
                gross_margin = 65
            elif carr < 10000000:
                arr_per_headcount = 200000
                magic_number = 0.5
                gross_margin = 75
            else:
                arr_per_headcount = 250000
                magic_number = 0.7
                gross_margin = 80
        
        # Calculate headcount with realistic bounds
        headcount = max(1, min(500, int(carr / arr_per_headcount))) if arr_per_headcount > 0 else max(1, int(carr / 100000))
        
        # Calculate sales & marketing with realistic bounds
        if magic_number > 0:
            sales_marketing = net_new_arr / magic_number
        else:
            sales_marketing = carr * 0.4  # 40% of ARR is more realistic
        
        # Ensure sales & marketing is reasonable
        sales_marketing = max(carr * 0.2, min(carr * 0.8, sales_marketing))
        
        # Calculate cash burn
        burn_multiple = (burn_multiple + growth_burn_multiple) / 2
        cash_burn = -net_new_arr * burn_multiple if burn_multiple > 0 else -carr * 0.3
        
        # Calculate customers (more realistic ratio)
        customers = max(5, int(headcount * 1.5))  # More realistic customer:employee ratio
        
        # Build complete feature set with realistic bounds
        # Ensure gross margin is realistic (60-85% for SaaS)
        gross_margin = max(60, min(85, gross_margin))
        
        inferred_metrics = {
            'cARR': carr,
            'Net New ARR': net_new_arr,
            'ARR YoY Growth (in %)': arr_yoy_growth,
            'Revenue YoY Growth (in %)': revenue_growth,
            'Gross Margin (in %)': gross_margin,
            'Sales & Marketing': sales_marketing,
            'Cash Burn (OCF & ICF)': cash_burn,
            'Headcount (HC)': headcount,
            'Customers (EoP)': customers,
            'Expansion & Upsell': net_new_arr * 0.3,  # Estimate
            'Churn & Reduction': -net_new_arr * 0.1,  # Estimate
            'EBITDA': carr * (gross_margin/100) * 0.2,  # Estimate
            'LTM Rule of 40% (ARR)': arr_yoy_growth + (gross_margin * 0.2),  # Estimate
            'Quarter Num': primary_inputs.get('Quarter Num', 1)
        }
        
        return inferred_metrics
    
    def guided_input_workflow(self) -> Dict:
        """
        Interactive guided input workflow.
        
        Returns:
            Dictionary with complete feature set
        """
        print("\n🎯 GUIDED INPUT SYSTEM")
        print("=" * 50)
        print("I'll help you create a forecast by asking for just a few key metrics.")
        print("I'll intelligently estimate the rest based on patterns from similar companies.\n")
        
        # Step 1: Get company name
        company_name = input("📝 Company name: ").strip() or "Your Company"
        
        # Step 2: Get current ARR (most critical)
        while True:
            try:
                carr_input = input("💰 Current ARR (Annual Recurring Revenue) in USD: ").strip()
                carr = float(carr_input.replace(',', '').replace('$', ''))
                break
            except ValueError:
                print("❌ Please enter a valid number (e.g., 1000000)")
        
        # Step 3: Get Net New ARR (second most critical)
        while True:
            try:
                net_new_input = input("📈 Net New ARR (new ARR added this quarter) in USD: ").strip()
                net_new_arr = float(net_new_input.replace(',', '').replace('$', ''))
                break
            except ValueError:
                print("❌ Please enter a valid number (e.g., 100000)")
        
        # Step 4: Get growth rate (optional, will be calculated if not provided)
        growth_input = input("📊 ARR YoY Growth rate in % (press Enter to calculate): ").strip()
        if growth_input:
            try:
                arr_yoy_growth = float(growth_input.replace('%', ''))
            except ValueError:
                print("⚠️ Invalid growth rate, will calculate from Net New ARR")
                arr_yoy_growth = (net_new_arr / carr) * 100 if carr > 0 else 0
        else:
            # Calculate growth rate from Net New ARR
            arr_yoy_growth = (net_new_arr / carr) * 100 if carr > 0 else 0
            print(f"📊 Calculated growth rate: {arr_yoy_growth:.1f}%")
        
        # Step 5: Ask about advanced mode
        advanced_mode = input("\n🔧 Would you like to enter advanced metrics? (y/N): ").strip().lower() == 'y'
        
        # Build primary inputs
        primary_inputs = {
            'cARR': carr,
            'Net New ARR': net_new_arr,
            'ARR YoY Growth (in %)': arr_yoy_growth,
            'Quarter Num': 1
        }
        
        # Infer secondary metrics
        inferred_metrics = self.infer_secondary_metrics(primary_inputs)
        
        # Show inferred metrics
        print(f"\n📊 INFERRED METRICS FOR {company_name.upper()}")
        print("=" * 50)
        print(f"Headcount: {inferred_metrics['Headcount (HC)']:,} employees")
        print(f"Sales & Marketing Spend: ${inferred_metrics['Sales & Marketing']:,.0f}")
        print(f"Cash Burn: ${inferred_metrics['Cash Burn (OCF & ICF)']:,.0f}")
        print(f"Gross Margin: {inferred_metrics['Gross Margin (in %)']:.1f}%")
        print(f"Customers: {inferred_metrics['Customers (EoP)']:,}")
        
        # Advanced mode
        if advanced_mode:
            print(f"\n🔧 ADVANCED MODE - Override any metrics:")
            print("(Press Enter to keep inferred value)")
            
            # Allow override of key metrics
            for metric, value in inferred_metrics.items():
                if metric in ['cARR', 'Net New ARR', 'ARR YoY Growth (in %)', 'Quarter Num']:
                    continue  # Skip primary inputs
                
                override = input(f"{metric}: {value:,.0f} → ").strip()
                if override:
                    try:
                        inferred_metrics[metric] = float(override)
                    except ValueError:
                        print(f"⚠️ Invalid input, keeping {value:,.0f}")
        
        # Add company name
        inferred_metrics['id_company'] = company_name
        inferred_metrics['Financial Quarter'] = 'FY24 Q1'  # Default quarter
        
        return inferred_metrics
    
    def create_forecast_input(self, guided_inputs: Dict) -> pd.DataFrame:
        """
        Create a properly formatted DataFrame for forecasting with historical data.
        
        Args:
            guided_inputs: Dictionary from guided input workflow
            
        Returns:
            DataFrame ready for model prediction with historical quarters
        """
        # Create historical data for the past 8 quarters (2 years)
        quarters = []
        base_arr = guided_inputs['cARR']
        base_growth = guided_inputs['ARR YoY Growth (in %)'] / 100  # Convert to decimal
        
        # Create historical quarters (FY23 Q1 to FY24 Q4)
        for year in [23, 24]:
            for quarter in [1, 2, 3, 4]:
                quarter_data = guided_inputs.copy()
                quarter_data['Financial Quarter'] = f'FY{year} Q{quarter}'
                quarter_data['Quarter Num'] = quarter
                
                # Calculate historical ARR based on growth rate
                if year == 23:
                    # FY23: Calculate backwards from current ARR
                    quarters_back = (24 - year) * 4 + (4 - quarter)
                    historical_arr = base_arr / ((1 + base_growth) ** quarters_back)
                else:
                    # FY24: Use current ARR for Q4, calculate others
                    if quarter == 4:
                        historical_arr = base_arr
                    else:
                        quarters_back = 4 - quarter
                        historical_arr = base_arr / ((1 + base_growth) ** quarters_back)
                
                quarter_data['cARR'] = historical_arr
                quarter_data['Net New ARR'] = historical_arr * base_growth if quarter > 1 else 0
                
                # Adjust other metrics proportionally
                if 'Headcount (HC)' in quarter_data:
                    quarter_data['Headcount (HC)'] = max(1, int(quarter_data['Headcount (HC)'] * (historical_arr / base_arr) ** 0.5))
                if 'Sales & Marketing' in quarter_data:
                    quarter_data['Sales & Marketing'] = quarter_data['Sales & Marketing'] * (historical_arr / base_arr)
                if 'Cash Burn (OCF & ICF)' in quarter_data:
                    quarter_data['Cash Burn (OCF & ICF)'] = quarter_data['Cash Burn (OCF & ICF)'] * (historical_arr / base_arr)
                
                quarters.append(quarter_data)
        
        # Create DataFrame
        df = pd.DataFrame(quarters)
        
        # Ensure all required columns are present
        required_columns = [
            'id_company', 'Financial Quarter', 'cARR', 'ARR YoY Growth (in %)',
            'Revenue YoY Growth (in %)', 'Gross Margin (in %)', 'EBITDA',
            'Cash Burn (OCF & ICF)', 'LTM Rule of 40% (ARR)', 'Sales & Marketing',
            'Headcount (HC)', 'Customers (EoP)', 'Expansion & Upsell',
            'Churn & Reduction', 'Quarter Num', 'Net New ARR'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Default value
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize the guided input system
    guided_system = GuidedInputSystem()
    
    # Run guided input workflow
    inputs = guided_system.guided_input_workflow()
    
    # Create forecast-ready DataFrame
    forecast_df = guided_system.create_forecast_input(inputs)
    
    print(f"\n✅ FORECAST INPUT READY")
    print("=" * 50)
    print(forecast_df.to_string(index=False)) 