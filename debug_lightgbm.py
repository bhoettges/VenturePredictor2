#!/usr/bin/env python3
"""
Debug script to test LightGBM model step by step
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def debug_lightgbm():
    print("🔍 Starting LightGBM debug...")
    
    # Step 1: Load the model
    try:
        from financial_prediction import load_trained_model
        print("✅ Import successful")
        
        trained_model = load_trained_model('lightgbm_financial_model.pkl')
        if trained_model:
            print("✅ Model loaded successfully")
            print(f"✅ Model type: {type(trained_model)}")
        else:
            print("❌ Model is None")
            return
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 2: Load and process test data
    try:
        df = pd.read_csv('test_company_2024.csv')
        print(f"✅ CSV loaded: {len(df)} rows")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Process data like the API does
        df_processed = df.copy()
        
        # Rename columns to match expected format
        column_mapping = {
            'ARR_End_of_Quarter': 'cARR',
            'Quarterly_Net_New_ARR': 'Net New ARR',
            'QRR_Quarterly_Recurring_Revenue': 'QRR',
            'Headcount': 'Headcount (HC)',
            'Gross_Margin_Percent': 'Gross Margin (in %)',
            'Net_Profit_Loss_Margin_Percent': 'Net_Profit_Loss_Margin_Percent'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed[new_col] = df_processed[old_col]
        
        # Add required fields
        if 'ARR YoY Growth (in %)' not in df_processed.columns:
            df_processed['ARR YoY Growth (in %)'] = df_processed['cARR'].pct_change() * 100
        
        if 'Revenue YoY Growth (in %)' not in df_processed.columns:
            df_processed['Revenue YoY Growth (in %)'] = df_processed['QRR'].pct_change() * 100
        
        # Fill missing required fields
        required_fields = [
            "ARR YoY Growth (in %)", "Revenue YoY Growth (in %)", "Gross Margin (in %)",
            "EBITDA", "Cash Burn (OCF & ICF)", "LTM Rule of 40% (ARR)", "Quarter Num"
        ]
        
        for field in required_fields:
            if field not in df_processed.columns:
                if field == 'EBITDA':
                    df_processed[field] = df_processed['cARR'] * 0.2
                elif field == 'Cash Burn (OCF & ICF)':
                    df_processed[field] = -df_processed['cARR'] * 0.3
                elif field == 'LTM Rule of 40% (ARR)':
                    df_processed[field] = df_processed['ARR YoY Growth (in %)'] + df_processed['Gross Margin (in %)'] * 0.2
                elif field == 'Quarter Num':
                    df_processed[field] = range(1, len(df_processed) + 1)
                else:
                    df_processed[field] = 0
        
        # Add company info
        df_processed['id_company'] = 'Test Company'
        df_processed['Financial Quarter'] = [f'FY24 Q{i}' for i in range(1, len(df_processed) + 1)]
        
        print(f"✅ Data processed: {len(df_processed)} rows")
        print(f"✅ Processed columns: {list(df_processed.columns)}")
        print(f"✅ Sample data:")
        print(df_processed.head())
        
    except Exception as e:
        print(f"❌ Failed to process data: {e}")
        return
    
    # Step 3: Try to make prediction
    try:
        from financial_prediction import predict_future_arr
        
        print(f"🔍 Attempting LightGBM prediction with {len(df_processed)} rows")
        print(f"🔍 Forecast DataFrame columns: {list(df_processed.columns)}")
        print(f"🔍 First row sample: {df_processed.iloc[0].to_dict()}")
        
        forecast_results = predict_future_arr(trained_model, df_processed)
        print(f"✅ LightGBM prediction successful!")
        print(f"✅ Result type: {type(forecast_results)}")
        print(f"✅ Result: {forecast_results}")
        
    except Exception as e:
        print(f"❌ LightGBM prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lightgbm()
