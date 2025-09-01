#!/usr/bin/env python3
"""
Deployment validation script for Lovable integration
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_imports():
    """Check if all required modules can be imported."""
    print("\n🔍 Checking Python imports...")
    
    required_modules = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'lightgbm',
        'sklearn',
        'langchain',
        'openai'
    ]
    
    all_good = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    return all_good

def check_model_files():
    """Check if model files exist."""
    print("\n🤖 Checking model files...")
    
    model_files = [
        ('lightgbm_financial_model.pkl', 'LightGBM Model'),
        ('202402_Copy.csv', 'Training Data'),
        ('gpt_info.json', 'GPT Info'),
    ]
    
    all_good = True
    for filepath, description in model_files:
        if check_file_exists(filepath, description):
            # Check file size
            size = os.path.getsize(filepath)
            print(f"   📊 Size: {size:,} bytes")
        else:
            all_good = False
    
    return all_good

def check_config_files():
    """Check if configuration files exist."""
    print("\n⚙️ Checking configuration files...")
    
    config_files = [
        ('requirements.txt', 'Python Dependencies'),
        ('render.yaml', 'Render Deployment Config'),
        ('README.md', 'Documentation'),
        ('fastapi_app.py', 'Main API Application'),
    ]
    
    all_good = True
    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good

def validate_api_structure():
    """Validate the API structure."""
    print("\n🔧 Validating API structure...")
    
    try:
        # Try to import the main app
        from fastapi_app import app, VALID_SECTORS, VALID_COUNTRIES, VALID_CURRENCIES
        
        print("✅ FastAPI app imported successfully")
        print(f"✅ Valid sectors: {len(VALID_SECTORS)} options")
        print(f"✅ Valid countries: {len(VALID_COUNTRIES)} options")
        print(f"✅ Valid currencies: {len(VALID_CURRENCIES)} options")
        
        # Check if endpoints are defined
        routes = [route.path for route in app.routes]
        required_routes = ['/', '/guided_forecast', '/chat', '/predict_csv', '/makro-analysis']
        
        for route in required_routes:
            if route in routes:
                print(f"✅ Endpoint: {route}")
            else:
                print(f"❌ Missing endpoint: {route}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ API validation failed: {e}")
        return False

def check_environment():
    """Check environment setup."""
    print("\n🌍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major == 3 and python_version.minor >= 10:
        print("✅ Python version is compatible")
    else:
        print("❌ Python version should be 3.10+")
        return False
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"✅ Current directory: {current_dir}")
    
    return True

def generate_deployment_summary():
    """Generate a deployment summary."""
    print("\n📋 DEPLOYMENT SUMMARY")
    print("=" * 50)
    
    summary = {
        "status": "READY FOR LOVABLE",
        "features": {
            "basic_mode": "Minimal inputs (company_name, current_arr, net_new_arr)",
            "enhanced_mode": "Optional sector/country/currency selection",
            "historical_arr": "Optional 4-quarter historical data",
            "advanced_mode": "Optional 14 key metrics override",
            "uncertainty": "±10% uncertainty bands",
            "chat_interface": "Conversational AI with LangChain",
            "model": "LightGBM only (removed XGBoost/Random Forest)"
        },
        "endpoints": [
            "GET / - API documentation",
            "POST /guided_forecast - Main forecasting",
            "POST /chat - Conversational AI",
            "POST /predict_csv - CSV upload",
            "GET /makro-analysis - Market indicators"
        ],
        "validation": {
            "sectors": "7 main sectors + 'Other'",
            "countries": "4 main countries + 'Other'",
            "currencies": "5 main currencies + 'Other'"
        },
        "deployment": {
            "platform": "Render (configured)",
            "python_version": "3.10.18",
            "cors": "Configured for web frontend",
            "environment_vars": "OPENAI_API_KEY required for chat"
        }
    }
    
    print(json.dumps(summary, indent=2))
    
    # Save summary to file
    with open('deployment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Deployment summary saved to: deployment_summary.json")

def main():
    """Run the complete deployment validation."""
    print("🚀 LOVABLE DEPLOYMENT VALIDATION")
    print("=" * 60)
    
    checks = [
        ("Environment", check_environment),
        ("Configuration Files", check_config_files),
        ("Model Files", check_model_files),
        ("Python Imports", check_imports),
        ("API Structure", validate_api_structure),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name} Check:")
        print("-" * 30)
        if not check_func():
            all_passed = False
            print(f"❌ {check_name} check failed!")
        else:
            print(f"✅ {check_name} check passed!")
    
    if all_passed:
        print(f"\n🎉 ALL CHECKS PASSED!")
        print(f"🚀 READY FOR LOVABLE DEPLOYMENT")
        generate_deployment_summary()
        
        print(f"\n📋 Next Steps:")
        print(f"1. Push code to your repository")
        print(f"2. Deploy to Render (or your preferred platform)")
        print(f"3. Set OPENAI_API_KEY environment variable")
        print(f"4. Test endpoints with Lovable frontend")
        print(f"5. Monitor performance and scale if needed")
        
    else:
        print(f"\n❌ DEPLOYMENT VALIDATION FAILED")
        print(f"Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main()
