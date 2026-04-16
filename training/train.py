"""
Training script for traffic signal RL agent
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_signal_env import TrafficSignalEnv
from agents.dqn_agent import TrafficSignalAgent
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train Traffic Signal RL Agent")
    parser.add_argument("--config", type=str, 
                       default="simulation/networks/simple/sumo_config.sumocfg",
                       help="Path to SUMO configuration file")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total timesteps for training")
    parser.add_argument("--render", action="store_true",
                       help="Render during training")
    
    args, _ = parser.parse_known_args()  # Ignore unknown args (e.g. 'train' from main.py)
    
    # Create environment
    env = TrafficSignalEnv(
        sumo_config=args.config,
        render_mode="human" if args.render else None
    )
    
    # Create and train agent
    agent = TrafficSignalAgent(env)
    agent.train(total_timesteps=args.timesteps)
    
    print("Training completed!")
    env.close()

if __name__ == "__main__":
    main()
