"""
Unit tests for the traffic environment
"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_signal_env import TrafficSignalEnv

class TestTrafficEnv(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.config_path = "simulation/networks/simple/sumo_config.sumocfg"
        
    def test_env_creation(self):
        """Test environment creation"""
        env = TrafficSignalEnv(
            sumo_config=self.config_path,
            max_steps=100
        )
        self.assertIsNotNone(env)
        
    def test_action_space(self):
        """Test action space dimensions"""
        env = TrafficSignalEnv(
            sumo_config=self.config_path,
            max_steps=100
        )
        self.assertEqual(env.action_space.n, 4)
        
    def test_observation_space(self):
        """Test observation space shape"""
        env = TrafficSignalEnv(
            sumo_config=self.config_path,
            max_steps=100
        )
        self.assertEqual(env.observation_space.shape, (6,))

if __name__ == '__main__':
    unittest.main()
