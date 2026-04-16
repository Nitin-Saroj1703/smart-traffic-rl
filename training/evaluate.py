"""
Evaluation script for trained traffic signal RL agent
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_signal_env import TrafficSignalEnv
from agents.dqn_agent import TrafficSignalAgent
from utils.helpers import MetricsLogger
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Traffic Signal RL Agent")
    parser.add_argument("--config", type=str,
                        default="simulation/networks/simple/sumo_config.sumocfg",
                        help="Path to SUMO configuration file")
    parser.add_argument("--model", type=str,
                        default="models/dqn_traffic",
                        help="Path to trained model file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render using SUMO-GUI during evaluation")

    args, _ = parser.parse_known_args()  # Ignore unknown args (e.g. 'evaluate' from main.py)

    # Create environment
    env = TrafficSignalEnv(
        sumo_config=args.config,
        render_mode="human" if args.render else None
    )

    # Load the trained agent
    agent = TrafficSignalAgent(env, model_path=args.model)
    agent.load()

    logger = MetricsLogger(log_dir="logs/evaluation")

    print(f"\nEvaluating model for {args.episodes} episodes...\n")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        total_wait = 0.0
        total_co2 = 0.0
        steps = 0

        while True:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_wait += info.get("total_waiting_time", 0)
            total_co2 += info.get("total_co2", 0)
            steps += 1

            if terminated or truncated:
                break

        avg_wait = total_wait / max(steps, 1)
        avg_co2 = total_co2 / max(steps, 1)
        avg_queue = -total_reward / max(steps, 1)  # Approximate from reward

        logger.log_episode(episode, total_reward, avg_queue, avg_wait, avg_co2)

        print(f"Episode {episode + 1:>2}/{args.episodes} | "
              f"Reward: {total_reward:>10.2f} | "
              f"Avg Wait: {avg_wait:>8.2f}s | "
              f"Avg CO2: {avg_co2:>10.2f}")

    logger.save()
    env.close()
    print("\nEvaluation completed! Results saved to logs/evaluation/")


if __name__ == "__main__":
    main()
