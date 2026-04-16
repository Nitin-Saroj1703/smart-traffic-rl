"""
Helper functions for the traffic signal control system
"""
import numpy as np
import pandas as pd
import json
import yaml
from typing import Dict, List, Any
import os
from datetime import datetime

class MetricsLogger:
    """Logs and tracks metrics during training/evaluation"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {
            'episode': [],
            'reward': [],
            'queue_length': [],
            'wait_time': [],
            'co2_emissions': [],
            'timestamp': []
        }
    
    def log_episode(self, episode: int, reward: float, 
                   queue_length: float, wait_time: float, co2: float):
        """Log metrics for one episode"""
        self.metrics['episode'].append(episode)
        self.metrics['reward'].append(reward)
        self.metrics['queue_length'].append(queue_length)
        self.metrics['wait_time'].append(wait_time)
        self.metrics['co2_emissions'].append(co2)
        self.metrics['timestamp'].append(datetime.now().isoformat())
    
    def save(self, filename: str = None):
        """Save metrics to CSV"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.log_dir, filename), index=False)
        print(f"Metrics saved to {self.log_dir}/{filename}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format")

def calculate_emissions_reduction(baseline_co2: float, current_co2: float) -> float:
    """Calculate percentage reduction in CO2 emissions"""
    if baseline_co2 == 0:
        return 0.0
    return ((baseline_co2 - current_co2) / baseline_co2) * 100

def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
    """Calculate exponential moving average"""
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema
