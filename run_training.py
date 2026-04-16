#!/usr/bin/env python3
"""
Quick training script with default parameters
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_single import main

if __name__ == "__main__":
    print("Smart Traffic RL - Quick Training")
    print("="*60)
    print("\nStarting PPO training with curriculum learning...")
    print("This may take 30-60 minutes depending on your hardware.\n")
    
    # Run main training pipeline
    main()
