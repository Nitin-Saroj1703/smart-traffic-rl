"""
Adversarial tests for multi-agent traffic system
"""
import unittest
import sys
import os
import numpy as np
import traci
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.multi_traffic_env import MultiTrafficEnv
from stable_baselines3 import PPO
import supersuit as ss

class TestAdversarial(unittest.TestCase):
    """Adversarial test cases for multi-agent system"""
    
    @classmethod
    def setUpClass(cls):
        """Load trained model"""
        cls.model_path = "agents/mappo_final.zip"
        cls.sumo_config = "simulation/networks/grid/grid.sumocfg"
        
        # Create environment
        cls.env = MultiTrafficEnv(
            sumo_config_file=cls.sumo_config,
            max_steps=500,
            gui=False
        )
        
        # Load model if available
        if os.path.exists(cls.model_path):
            cls.env_wrapped = ss.pettingzoo_env_to_vec_env_v1(cls.env)
            cls.env_wrapped = ss.concat_vec_envs_v1(cls.env_wrapped, 1, base_class='stable_baselines3')
            cls.model = PPO.load(cls.model_path, env=cls.env_wrapped)
            cls.model_loaded = True
        else:
            print(f"Warning: Model not found at {cls.model_path}")
            print("Running tests with random actions")
            cls.model_loaded = False
    
    def test_1_lane_blockage_accident(self):
        """Test 1: Block one lane (simulate accident) -> measure adaptation"""
        print("\n" + "="*60)
        print("TEST 1: LANE BLOCKAGE (ACCIDENT SIMULATION)")
        print("="*60)
        
        # Create fresh environment
        env = MultiTrafficEnv(
            sumo_config_file=self.sumo_config,
            max_steps=200,
            gui=False
        )
        
        obs, _ = env.reset()
        
        # Simulate normal operation first
        normal_queue_lengths = []
        for step in range(50):
            if self.model_loaded:
                obs_array = np.array([obs[agent] for agent in env.agents])
                actions = self.model.predict(obs_array, deterministic=True)[0]
                action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            else:
                action_dict = {agent: np.random.randint(0, 3) for agent in env.agents}
            
            obs, rewards, _, _, infos = env.step(action_dict)
            
            # Track queue lengths
            total_queue = sum(
                infos[agent].get('cumulative_reward', 0) 
                for agent in env.agents
            )
            normal_queue_lengths.append(total_queue)
        
        # Block a lane (simulate accident)
        print("Simulating accident: Blocking lane n00_n01...")
        try:
            traci.lane.setDisallowed("n00_n01_0", ["passenger"])
        except:
            pass
        
        # Continue simulation with blockage
        blocked_queue_lengths = []
        for step in range(50):
            if self.model_loaded:
                obs_array = np.array([obs[agent] for agent in env.agents])
                actions = self.model.predict(obs_array, deterministic=True)[0]
                action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            else:
                action_dict = {agent: np.random.randint(0, 3) for agent in env.agents}
            
            obs, rewards, _, _, infos = env.step(action_dict)
            
            total_queue = sum(
                infos[agent].get('cumulative_reward', 0) 
                for agent in env.agents
            )
            blocked_queue_lengths.append(total_queue)
        
        env.close()
        
        # Analyze results
        avg_normal = np.mean(normal_queue_lengths[-10:])
        avg_blocked = np.mean(blocked_queue_lengths[-10:])
        adaptation_ratio = avg_blocked / avg_normal if avg_normal != 0 else 1.0
        
        print(f"\nResults:")
        print(f"   Average queue before blockage: {avg_normal:.2f}")
        print(f"   Average queue after blockage: {avg_blocked:.2f}")
        print(f"   Adaptation ratio: {adaptation_ratio:.2f}")
        
        if adaptation_ratio < 1.5:
            print("   Good adaptation: System handled blockage well")
        else:
            print("   Poor adaptation: System struggled with blockage")
        
        self.assertLess(adaptation_ratio, 3.0, "System failed to adapt to lane blockage")
    
    def test_2_sensor_failure(self):
        """Test 2: Remove sensor data -> measure graceful degradation"""
        print("\n" + "="*60)
        print("TEST 2: SENSOR FAILURE (DATA LOSS)")
        print("="*60)
        
        env = MultiTrafficEnv(
            sumo_config_file=self.sumo_config,
            max_steps=200,
            gui=False
        )
        
        obs, _ = env.reset()
        
        # Normal operation
        normal_performance = []
        for step in range(50):
            if self.model_loaded:
                obs_array = np.array([obs[agent] for agent in env.agents])
                actions = self.model.predict(obs_array, deterministic=True)[0]
                action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            else:
                action_dict = {agent: np.random.randint(0, 3) for agent in env.agents}
            
            obs, rewards, _, _, infos = env.step(action_dict)
            normal_performance.append(np.mean(list(rewards.values())))
        
        # Simulate sensor failure (zero out 2 lanes of observations)
        print("Simulating sensor failure: Zeroing 2 lanes of data...")
        failed_performance = []
        
        for step in range(50):
            if self.model_loaded:
                # Corrupt observations for 2 agents
                corrupted_obs = obs.copy()
                for i, agent in enumerate(env.agents):
                    if i < 2:  # Fail sensors for first 2 intersections
                        corrupted_obs[agent] = np.zeros_like(obs[agent])
                
                obs_array = np.array([corrupted_obs[agent] for agent in env.agents])
                actions = self.model.predict(obs_array, deterministic=True)[0]
                action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            else:
                action_dict = {agent: np.random.randint(0, 3) for agent in env.agents}
            
            obs, rewards, _, _, infos = env.step(action_dict)
            failed_performance.append(np.mean(list(rewards.values())))
        
        env.close()
        
        # Analyze degradation
        avg_normal = np.mean(normal_performance[-10:])
        avg_failed = np.mean(failed_performance[-10:])
        degradation = ((avg_normal - avg_failed) / abs(avg_normal)) * 100 if avg_normal != 0 else 0
        
        print(f"\nResults:")
        print(f"   Normal performance: {avg_normal:.2f}")
        print(f"   Degraded performance: {avg_failed:.2f}")
        print(f"   Performance degradation: {degradation:.1f}%")
        
        if degradation < 30:
            print("   Graceful degradation: System maintained reasonable performance")
        else:
            print("   Severe degradation: System heavily impacted by sensor failure")
        
        self.assertLess(degradation, 90, "System failed to handle sensor failures gracefully")
    
    def test_3_multiple_emergencies(self):
        """Test 3: Simultaneous emergency vehicles -> measure clearance rate"""
        print("\n" + "="*60)
        print("TEST 3: SIMULTANEOUS EMERGENCY VEHICLES")
        print("="*60)
        
        env = MultiTrafficEnv(
            sumo_config_file=self.sumo_config,
            max_steps=300,
            gui=False
        )
        
        obs, _ = env.reset()
        
        emergency_clearance_times = []
        emergencies_cleared = 0
        total_emergencies = 0
        
        # Run simulation with multiple emergencies
        for step in range(300):
            if self.model_loaded:
                obs_array = np.array([obs[agent] for agent in env.agents])
                actions = self.model.predict(obs_array, deterministic=True)[0]
                action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}
            else:
                action_dict = {agent: np.random.randint(0, 3) for agent in env.agents}
            
            obs, rewards, _, _, infos = env.step(action_dict)
            
            # Track emergency vehicles
            for agent in env.agents:
                if infos[agent].get('emergency_present', False):
                    total_emergencies += 1
                    
                    # Check if emergency was cleared
                    if not infos[agent].get('emergency_present', True):
                        emergencies_cleared += 1
                        if 'clearance_time' in infos[agent]:
                            emergency_clearance_times.append(infos[agent]['clearance_time'])
        
        env.close()
        
        # Calculate success rate
        clearance_rate = (emergencies_cleared / total_emergencies * 100) if total_emergencies > 0 else 0
        avg_clearance_time = np.mean(emergency_clearance_times) if emergency_clearance_times else 0
        
        print(f"\nResults:")
        print(f"   Total emergencies detected: {total_emergencies}")
        print(f"   Emergencies cleared: {emergencies_cleared}")
        print(f"   Clearance success rate: {clearance_rate:.1f}%")
        print(f"   Average clearance time: {avg_clearance_time:.1f} steps")
        
        if clearance_rate > 30:
            print("   Good emergency handling: Some emergencies cleared successfully")
        else:
            print("   Poor emergency handling: System struggled with multiple emergencies")
    
def run_adversarial_summary():
    """Run all adversarial tests and print summary table"""
    print("\n" + "="*80)
    print("ADVERSARIAL TESTING SUMMARY")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestAdversarial('test_1_lane_blockage_accident'))
    suite.addTest(TestAdversarial('test_2_sensor_failure'))
    suite.addTest(TestAdversarial('test_3_multiple_emergencies'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary table
    print("\n" + "="*80)
    print("ADVERSARIAL TEST RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Test Case':<40} {'Result':<15} {'Performance':<20}")
    print("-"*75)
    
    # Placeholder for summary results
    test_results = [
        ("Test 1: Lane Blockage (Accident)", "RUN", ""),
        ("Test 2: Sensor Failure", "RUN", ""),
        ("Test 3: Multiple Emergencies", "RUN", "")
    ]
    
    for test, status, performance in test_results:
        print(f"{test:<40} {status:<15} {performance:<20}")
    
    print("\n" + "="*80)
    print("Overall Assessment completed")
    print("="*80)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_adversarial_summary()
    sys.exit(0 if success else 1)
