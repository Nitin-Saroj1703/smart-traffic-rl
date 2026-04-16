#!/usr/bin/env python3
"""
Smart Traffic RL - Main Entry Point
Complete reinforcement learning system for traffic signal control
"""
import sys
import os
import argparse
import subprocess
from typing import Dict, Any
import io

# Force UTF-8 encoding for console output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure SUMO bin is in path for traci to find it on Windows
sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
if sumo_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

from config import (
    sumo_config,
    env_config,
    training_config,
    dashboard_config,
    paths_config
)

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        SMART TRAFFIC RL - Intelligent Traffic Control        ║
    ║                                                              ║
    ║     Multi-Agent Reinforcement Learning for Urban Mobility    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu options"""
    menu = """
    ┌─────────────────────────────────────────────────────────────┐
    │                       MAIN MENU                             │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Train Single Agent (PPO with Curriculum Learning)       │
    │  2. Train Multi-Agent (MAPPO for 9 Intersections)           │
    │  3. Run Dashboard (Real-time Visualization)                 │
    │  4. Run Evaluation Only                                     │
    │  5. Run Adversarial Tests                                   │
    │  6. Generate SUMO Network                                   │
    │  7. Run All Tests                                           │
    │  8. Exit                                                    │
    └─────────────────────────────────────────────────────────────┘
    """
    print(menu)

def train_single_agent():
    """Train single agent with curriculum learning"""
    print("\n" + "="*60)
    print("TRAINING SINGLE AGENT (PPO with Curriculum Learning)")
    print("="*60)
    
    from training.train_single import main as train_single
    
    # Override with config values if needed
    print(f"\nConfiguration:")
    print(f"  - Stage 1: {training_config.STAGE_1_TIMESTEPS} steps, {training_config.STAGE_1_MAX_VEHICLES} vehicles")
    print(f"  - Stage 2: {training_config.STAGE_2_TIMESTEPS} steps, {training_config.STAGE_2_MAX_VEHICLES} vehicles")
    print(f"  - Stage 3: {training_config.STAGE_3_TIMESTEPS} steps, {training_config.STAGE_3_MAX_VEHICLES} vehicles")
    print(f"  - Learning rate: {training_config.LEARNING_RATE}")
    print(f"  - Model save path: {paths_config.SINGLE_AGENT_MODEL}")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() == 'y':
        train_single()
    else:
        print("Training cancelled.")

def train_multi_agent():
    """Train multi-agent system"""
    print("\n" + "="*60)
    print("TRAINING MULTI-AGENT (MAPPO for 9 Intersections)")
    print("="*60)
    
    from training.train_multi import main as train_multi
    
    print(f"\nConfiguration:")
    print(f"  - Agents: {sumo_config.NUM_INTERSECTIONS} intersections")
    print(f"  - Timesteps: {training_config.MAPPO_TOTAL_TIMESTEPS}")
    print(f"  - Local reward weight: {env_config.LOCAL_REWARD_WEIGHT}")
    print(f"  - Global reward weight: {env_config.GLOBAL_REWARD_WEIGHT}")
    print(f"  - Model save path: {paths_config.MULTI_AGENT_MODEL}")
    
    # Check if pretrained model exists
    if os.path.exists(paths_config.SINGLE_AGENT_MODEL):
        print(f"  Pretrained model found: {paths_config.SINGLE_AGENT_MODEL}")
    else:
        print(f"  No pretrained model found. Training from scratch.")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() == 'y':
        train_multi()
    else:
        print("Training cancelled.")

def run_dashboard():
    """Launch Streamlit dashboard"""
    print("\n" + "="*60)
    print("LAUNCHING DASHBOARD")
    print("="*60)
    
    port = dashboard_config.DEFAULT_PORT
    host = dashboard_config.DEFAULT_HOST
    
    print(f"\nDashboard will be available at: http://{host}:{port}")
    print("\nPress Ctrl+C to stop the dashboard.")
    print("\nLaunching...")
    
    try:
        # Run streamlit as a python module using the current python executable
        subprocess.run([
            sys.executable, "-m", "streamlit",
            "run", "dashboard/app.py",
            "--server.port", str(port),
            "--server.address", host
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nFailed to launch dashboard: {e}")
        print("\nMake sure streamlit is installed in the virtual environment.")

def run_evaluation():
    """Run evaluation only"""
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)
    
    print("\nSelect evaluation mode:")
    print("  1. Single Agent Evaluation")
    print("  2. Multi-Agent Evaluation")
    print("  3. Compare Both")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        from training.train_single import main as run_eval_single
        print("\nEvaluating single agent...")
        # Note: In a refactored training script, we'd call an evaluation function
        # For now, we reuse the training main context if it supports eval-only
        pass
    else:
        print("Evaluation mode not fully implemented yet in main menu.")

def run_adversarial_tests():
    """Run adversarial tests"""
    print("\n" + "="*60)
    print("RUNNING ADVERSARIAL TESTS")
    print("="*60)
    
    from tests.test_adversarial import run_adversarial_summary
    
    print("\nRunning tests:")
    print("  1. Lane Blockage (Accident Simulation)")
    print("  2. Sensor Failure (Data Loss)")
    print("  3. Multiple Emergency Vehicles")
    
    response = input("\nProceed with tests? (y/n): ")
    if response.lower() == 'y':
        success = run_adversarial_summary()
        if success:
            print("\nAll adversarial tests passed!")
        else:
            print("\nSome tests failed. Check output above.")
    else:
        print("Tests cancelled.")

def generate_sumo_network():
    """Generate SUMO network"""
    print("\n" + "="*60)
    print("GENERATING SUMO NETWORK")
    print("="*60)
    
    try:
        # Run the network generator script directly since the simulation
        # subdirectories don't have __init__.py for package imports
        import subprocess
        result = subprocess.run(
            [sys.executable, "simulation/networks/grid/generate_network.py"],
            capture_output=False
        )
        if result.returncode == 0:
            print("\nNetwork generated successfully!")
            print(f"   Network file: {paths_config.NETWORK_FILE}")
            print(f"   Routes file: {paths_config.ROUTES_FILE}")
            print(f"   SUMO config: {paths_config.SUMO_CONFIG}")
        else:
            print("\nNetwork generation failed!")
    except Exception as e:
        print(f"\nFailed to generate network: {e}")

def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("SUMO Tests", "tests/test_sumo.py"),
        ("Environment Tests", "tests/test_env.py"),
        ("Adversarial Tests", "tests/test_adversarial.py")
    ]
    
    results = {}
    
    # Use venv python on Windows
    python_exe = sys.executable
    
    for test_name, test_path in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        result = subprocess.run(
            [python_exe, test_path],
            capture_output=False
        )
        
        results[test_name] = result.returncode == 0
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check output above.")

def check_requirements():
    """Check if all requirements are installed"""
    print("\nChecking requirements...")
    
    required_packages = [
        'gymnasium',
        'stable_baselines3',
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'streamlit',
        'pettingzoo',
        'supersuit'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  OK: {package}")
        except ImportError:
            print(f"  MISSING: {package}")
            missing.append(package)
    
    # Check SUMO
    try:
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  OK: sumo {result.stdout.split()[1]}")
        else:
            print(f"  MISSING: sumo")
            missing.append('sumo')
    except Exception:
        print(f"  MISSING: sumo")
        missing.append('sumo')
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        return False
    
    print("\nAll core requirements satisfied!")
    return True

def main():
    """Main entry point"""
    # Check requirements first
    if not check_requirements():
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print_banner()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Smart Traffic RL System')
    parser.add_argument('--mode', choices=['train', 'multi', 'dashboard', 'eval', 'test'],
                       help='Run specific mode directly')
    args = parser.parse_args()
    
    # If mode specified, run directly
    if args.mode:
        mode_map = {
            'train': train_single_agent,
            'multi': train_multi_agent,
            'dashboard': run_dashboard,
            'eval': run_evaluation,
            'test': run_all_tests
        }
        mode_map[args.mode]()
        return
    
    # Interactive menu
    while True:
        print_menu()
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == '1':
            train_single_agent()
        elif choice == '2':
            train_multi_agent()
        elif choice == '3':
            run_dashboard()
        elif choice == '4':
            run_evaluation()
        elif choice == '5':
            run_adversarial_tests()
        elif choice == '6':
            generate_sumo_network()
        elif choice == '7':
            run_all_tests()
        elif choice == '8':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter 1-8.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
