#!/usr/bin/env python3
"""
Quick test script for the traffic environment
"""
import sys
import os
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_env import run_manual_test
import unittest

if __name__ == "__main__":
    print("Smart Traffic RL - Environment Test")
    print("="*50)
    
    # Check if SUMO is installed
    try:
        # Update PATH to include SUMO bin if not present
        sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
        if sumo_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")
            
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("SUMO is installed")
            print(f"   Version: {result.stdout.split()[1]}")
        else:
            print("SUMO is not properly installed")
            sys.exit(1)
    except FileNotFoundError:
        print("SUMO is not installed")
        print("   Please install SUMO first:")
        print("   Ubuntu: sudo apt-get install sumo sumo-tools")
        print("   macOS: brew install sumo")
        sys.exit(1)
    
    # Check if network exists
    network_path = "simulation/networks/grid/grid_network.net.xml"
    if not os.path.exists(network_path):
        print("\nNetwork file not found. Generating network...")
        # Point to the grid generator
        import subprocess
        subprocess.run([sys.executable, "simulation/networks/grid/generate_network.py"], check=True)
    else:
        print(f"Network file found at {network_path}")
    
    # Run manual test
    print("\n" + "="*50)
    success = run_manual_test()
    
    if success:
        print("\nAll tests passed! Environment is ready for training.")
    else:
        print("\nTests failed. Please check the errors above.")
        sys.exit(1)
