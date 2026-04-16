#!/usr/bin/env python3
"""
Run complete multi-agent training and testing pipeline
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Smart Traffic RL - Multi-Agent System")
    print("="*60)
    
    # Step 1: Train MAPPO
    print("\nStep 1: Training MAPPO on 9 intersections...")
    try:
        from training.train_multi import main as train_multi
        train_multi()
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    # Step 2: Run adversarial tests
    print("\nStep 2: Running adversarial tests...")
    try:
        from tests.test_adversarial import run_adversarial_summary
        success = run_adversarial_summary()
    except Exception as e:
        print(f"Error during adversarial tests: {e}")
        return False
    
    if success:
        print("\nMulti-agent system successfully trained and tested!")
        print("\nGenerated files:")
        print("   - agents/mappo_final.zip")
        print("   - training/results_multi/per_intersection_wait_times.png")
        print("   - training/results_multi/network_wide_reduction.png")
        print("   - training/results_multi/coordination_effects.png")
    else:
        print("\nSome tests failed. Check the output above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
