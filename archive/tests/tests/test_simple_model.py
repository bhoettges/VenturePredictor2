#!/usr/bin/env python3
"""
Test the simple single-quarter models with bias correction.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 TESTING SIMPLE SINGLE-QUARTER MODELS")
    print("=" * 60)
    
    # Load the single quarter models
    try:
        with open('lightgbm_single_quarter_models.pkl', 'rb') as f:
            models = pickle.load(f)
        print("✅ Single quarter models loaded successfully.")
    except FileNotFoundError:
        print("❌ ERROR: lightgbm_single_quarter_models.pkl not found.")
        return
    
    # Load test data
    try:
        test_data = pd.read_csv('test_company_2024.csv')
        print("✅ Test data loaded successfully.")
    except FileNotFoundError:
        print("❌ ERROR: test_company_2024.csv not found.")
        return
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Available columns: {list(test_data.columns)}")
    
    # Show what models we have
    if isinstance(models, dict):
        print(f"Available models: {list(models.keys())}")
    else:
        print("Models loaded as single object")

if __name__ == "__main__":
    main()
