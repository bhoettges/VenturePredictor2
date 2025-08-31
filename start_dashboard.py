#!/usr/bin/env python3
"""
Startup script for the Financial Forecasting Dashboard.
Runs the FastAPI backend only - frontend is handled by Loveable AI.
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI backend...")
    try:
        # Start the FastAPI server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "fastapi_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        return process
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def main():
    """Main function to start the backend server."""
    print("🎯 Financial Forecasting Dashboard - Backend Only")
    print("=" * 50)
    
    # Check if required files exist
    if not Path("fastapi_app.py").exists():
        print("❌ Error: fastapi_app.py not found!")
        return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    print("\n✅ Backend started successfully!")
    print("=" * 50)
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🌐 Frontend: Use Loveable AI to connect to this backend")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        backend_process.terminate()
        print("✅ Server stopped.")

if __name__ == "__main__":
    main() 