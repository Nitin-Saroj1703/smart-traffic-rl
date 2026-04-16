"""
Unit tests for the traffic environment
"""
import unittest
import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.traffic_env import TrafficEnv

class TestTrafficEnv(unittest.TestCase):
    """Test cases for TrafficEnv"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - check if SUMO config exists"""
        cls.sumo_config = "simulation/networks/grid/grid.sumocfg"
        if not os.path.exists(cls.sumo_config):
            print(f"⚠️ Warning: SUMO config not found at {cls.sumo_config}")
            print("Please run 'python simulation/networks/grid/generate_network.py' first")
    
    def setUp(self):
        """Set up test environment"""
        self.env = TrafficEnv(
            sumo_config_file=self.sumo_config,
            intersection_id="n11",
            max_steps=50,
            gui=False
        )
    
    def tearDown(self):
        """Clean up after test"""
        if hasattr(self, 'env'):
            self.env.close()
    
    def test_env_creation(self):
        """Test environment creation"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.action_space.n, 3)
        self.assertEqual(self.env.observation_space.shape, (18,))
        print("✅ Environment creation: PASSED")
    
    def test_observation_space(self):
        """Test observation space properties"""
        obs_space = self.env.observation_space
        
        # Check shape
        self.assertEqual(obs_space.shape, (18,))
        
        # Check bounds
        self.assertTrue(np.all(obs_space.low >= -np.inf))
        self.assertTrue(np.all(obs_space.high <= np.inf))
        
        print("✅ Observation space: PASSED")
    
    def test_action_space(self):
        """Test action space properties"""
        action_space = self.env.action_space
        
        # Check number of actions
        self.assertEqual(action_space.n, 3)
        
        # Check valid actions
        for action in [0, 1, 2]:
            self.assertTrue(action_space.contains(action))
        
        # Check invalid actions
        self.assertFalse(action_space.contains(-1))
        self.assertFalse(action_space.contains(3))
        
        print("✅ Action space: PASSED")
    
    def test_reset(self):
        """Test reset functionality"""
        obs, info = self.env.reset()
        
        # Check observation shape and type
        self.assertEqual(obs.shape, (18,))
        self.assertEqual(obs.dtype, np.float32)
        
        # Check observation values are within bounds
        self.assertTrue(np.all(obs >= self.env.observation_space.low))
        self.assertTrue(np.all(obs <= self.env.observation_space.high))
        
        print("✅ Reset: PASSED")
    
    def test_step(self):
        """Test step functionality"""
        obs, info = self.env.reset()
        
        # Test all three actions
        for action in [0, 1, 2]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Check return types
            self.assertIsInstance(obs, np.ndarray)
            self.assertIsInstance(reward, (float, np.float32, np.float64))
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIsInstance(info, dict)
            
            # Check observation
            self.assertEqual(obs.shape, (18,))
            
            # Check info contains expected keys
            self.assertIn('step', info)
            self.assertIn('total_queue', info)
            self.assertIn('total_co2', info)
            self.assertIn('episode_reward', info)
        
        print("✅ Step: PASSED")
    
    def test_random_steps(self):
        """Test 50 random steps and print rewards"""
        print("\n" + "="*60)
        print("Testing 50 random steps")
        print("="*60)
        
        obs, info = self.env.reset()
        
        rewards = []
        queue_lengths = []
        
        for step in range(50):
            # Take random action
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            rewards.append(reward)
            queue_lengths.append(info['total_queue'])
            
            # Print progress every 10 steps
            if (step + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                avg_queue = np.mean(queue_lengths[-10:])
                print(f"Step {step+1:3d}: "
                      f"Action={action}, "
                      f"Reward={reward:7.2f}, "
                      f"Avg Reward (last 10)={avg_reward:7.2f}, "
                      f"Queue={info['total_queue']:5.1f}, "
                      f"Emergency={info['emergency_present']}")
            
            if terminated or truncated:
                break
        
        # Print summary statistics
        print("\n" + "-"*60)
        print("Summary Statistics:")
        print(f"Total steps: {step+1}")
        print(f"Total reward: {sum(rewards):.2f}")
        print(f"Mean reward: {np.mean(rewards):.2f}")
        print(f"Std reward: {np.std(rewards):.2f}")
        print(f"Min reward: {np.min(rewards):.2f}")
        print(f"Max reward: {np.max(rewards):.2f}")
        print(f"Mean queue length: {np.mean(queue_lengths):.1f}")
        print("="*60)
        
        print("✅ Random steps test: PASSED")
    
    def test_observation_shape_consistency(self):
        """Test that observations always have correct shape"""
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (18,))
        
        for _ in range(20):
            action = self.env.action_space.sample()
            obs, _, _, _, _ = self.env.step(action)
            self.assertEqual(obs.shape, (18,))
        
        print("✅ Observation shape consistency: PASSED")

def run_manual_test():
    """Run a manual test with detailed output"""
    print("\n" + "="*80)
    print("MANUAL ENVIRONMENT TEST")
    print("="*80)
    
    # Check if SUMO config exists
    sumo_config = "simulation/networks/grid/grid.sumocfg"
    if not os.path.exists(sumo_config):
        print(f"SUMO config not found at {sumo_config}")
        print("Please run the following command first:")
        print("  python simulation/networks/grid/generate_network.py")
        return False
    
    try:
        # Create environment
        print("\nCreating TrafficEnv...")
        env = TrafficEnv(
            sumo_config_file=sumo_config,
            intersection_id="n11",
            max_steps=100,
            gui=False
        )
        
        print(f"Environment created")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Observation space: {env.observation_space}")
        
        # Reset environment
        print("\nResetting environment...")
        obs, info = env.reset()
        print(f"Environment reset")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Initial observation: {obs}")
        
        # Run steps
        print("\nRunning 50 random steps...")
        print("-" * 80)
        print(f"{'Step':<6} {'Action':<8} {'Reward':<10} {'Queue':<8} {'CO2':<10} {'Emergency':<10}")
        print("-" * 80)
        
        total_reward = 0
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Print step information
            print(f"{step+1:<6} {action:<8} {reward:<10.2f} "
                  f"{info['total_queue']:<8.1f} {info['total_co2']:<10.1f} "
                  f"{str(info['emergency_present']):<10}")
            
            if terminated:
                print(f"\nEpisode terminated at step {step+1}")
                break
            if truncated:
                print(f"\nEpisode truncated at step {step+1}")
                break
        
        print("-" * 80)
        print(f"\nFinal Statistics:")
        print(f"   - Total steps: {step+1}")
        print(f"   - Total reward: {total_reward:.2f}")
        print(f"   - Average reward per step: {total_reward/(step+1):.2f}")
        
        # Close environment
        env.close()
        print("\nEnvironment closed successfully")
        print("\n" + "="*80)
        print("MANUAL TEST PASSED")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First run manual test
    success = run_manual_test()
    
    if success:
        # Then run unit tests
        print("\nRunning unit tests...\n")
        unittest.main(argv=[''], verbosity=2, exit=False)
    else:
        print("\n❌ Manual test failed. Please fix issues before running unit tests.")
        sys.exit(1)
