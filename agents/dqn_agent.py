"""
DQN Agent for Traffic Signal Control
"""
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np
import os

class TrafficSignalAgent:
    """
    DQN-based agent for traffic signal control
    """
    def __init__(self, env, model_path: str = "models/dqn_traffic"):
        self.env = Monitor(env)
        self.model_path = model_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize DQN model
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            verbose=1,
            tensorboard_log="./tensorboard_logs/"
        )
    
    def train(self, total_timesteps: int = 100000):
        """Train the agent"""
        self.model.learn(total_timesteps=total_timesteps)
        self.save()
    
    def predict(self, observation):
        """Get action from the model"""
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def save(self):
        """Save the model"""
        self.model.save(self.model_path)
    
    def load(self):
        """Load the model"""
        self.model = DQN.load(self.model_path, env=self.env)
