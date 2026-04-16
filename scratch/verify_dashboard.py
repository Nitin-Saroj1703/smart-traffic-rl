# Quick verification script
import sys
import os

print("--- Verifying dashboard components ---")

# Check required files
required_files = [
    "dashboard/app.py",
    "dashboard/utils.py",
    "dashboard/config.yaml",
    "run_dashboard.py"
]

all_present = True
for file in required_files:
    if os.path.exists(file):
        print(f"[OK] {file}")
    else:
        print(f"[MISSING] {file}")
        all_present = False

if all_present:
    print("\nAll dashboard files present!")
    print("\nDashboard Pages:")
    print("   1. Live Simulation View")
    print("   2. Performance Comparison")
    print("   3. Agent Decision Explainer")
    print("   4. Configuration Panel")
    print("\nReady to launch!")
else:
    print("\nSome files are missing. Please check the dashboard directory.")
