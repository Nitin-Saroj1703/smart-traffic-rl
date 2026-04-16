#!/usr/bin/env python3
"""
Launch the Smart Traffic RL Dashboard
"""
import os
import sys
import subprocess

def main():
    print("Launching Smart Traffic RL Dashboard")
    print("="*50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("Streamlit is installed")
    except ImportError:
        print("Streamlit is not installed")
        print("Installing streamlit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    from config import paths_config, dashboard_config
    
    # Check if model exists
    model_path = paths_config.MULTI_AGENT_MODEL
    if not os.path.exists(model_path):
        print(f"\nWarning: Model not found at {model_path}")
        print("The dashboard will run in demo mode with simulated data.")
    
    # Launch dashboard
    print("\nLaunching dashboard at http://localhost:8501")
    print("Press Ctrl+C to stop\n")
    
    subprocess.run([
        "powershell", "-Command", 
        f"& .\\.venv\\Scripts\\streamlit run dashboard/app.py --server.port 8501 --server.address localhost"
    ])

if __name__ == "__main__":
    main()
