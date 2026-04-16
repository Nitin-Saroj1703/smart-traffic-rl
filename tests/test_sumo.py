"""
Test SUMO simulation setup and utilities
"""
import unittest
import sys
import os
import traci
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure SUMO bin is in path for traci to find it on Windows
sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
if sumo_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

from utils.sumo_utils import SUMOUtils
from config import paths_config, sumo_config

class TestSUMO(unittest.TestCase):
    """Test SUMO simulation functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class"""
        cls.sumo_config_path = paths_config.SUMO_CONFIG
        cls.sumo_utils = SUMOUtils()
        
        # Check if SUMO config exists
        if not os.path.exists(cls.sumo_config_path):
            raise unittest.SkipTest(f"SUMO config not found at {cls.sumo_config_path}")
    
    def setUp(self):
        """Start SUMO before each test"""
        sumo_binary = "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_config_path,
            "--no-warnings", "true",
            "--time-to-teleport", "-1"
        ]
        
        # Ensure SUMO bin is in path for traci
        sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
        if sumo_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")
        
        traci.start(sumo_cmd)
        
        # Run a few steps to initialize
        for _ in range(5):
            traci.simulationStep()
    
    def tearDown(self):
        """Close SUMO after each test"""
        try:
            traci.close()
        except:
            pass
    
    def test_simulation_starts(self):
        """Test that SUMO simulation starts correctly"""
        time = traci.simulation.getTime()
        self.assertGreaterEqual(time, 0)
        print(f"Simulation started at time {time}")
    
    def test_get_queue_length(self):
        """Test queue length calculation"""
        # Get a lane
        lanes = traci.lane.getIDList()
        if lanes:
            lane_id = lanes[0]
            queue = self.sumo_utils.get_queue_length(lane_id)
            self.assertIsInstance(queue, int)
            self.assertGreaterEqual(queue, 0)
            print(f"Queue length for {lane_id}: {queue}")
    
    def test_get_avg_speed(self):
        """Test average speed calculation"""
        lanes = traci.lane.getIDList()
        if lanes:
            lane_id = lanes[0]
            speed = self.sumo_utils.get_avg_speed(lane_id)
            self.assertIsInstance(speed, float)
            self.assertGreaterEqual(speed, 0)
            print(f"Average speed for {lane_id}: {speed:.2f} m/s")
    
    def test_get_co2_emission(self):
        """Test CO2 emission calculation"""
        lanes = traci.lane.getIDList()
        if lanes:
            lane_id = lanes[0]
            co2 = self.sumo_utils.get_co2_emission(lane_id)
            self.assertIsInstance(co2, float)
            self.assertGreaterEqual(co2, 0)
            print(f"CO2 emission for {lane_id}: {co2:.2f} mg/s")
    
    def test_get_waiting_time(self):
        """Test waiting time calculation"""
        lanes = traci.lane.getIDList()
        if lanes:
            lane_id = lanes[0]
            wait_time = self.sumo_utils.get_waiting_time(lane_id)
            self.assertIsInstance(wait_time, float)
            self.assertGreaterEqual(wait_time, 0)
            print(f"Waiting time for {lane_id}: {wait_time:.2f} s")
    
    def test_detect_emergency_vehicle(self):
        """Test emergency vehicle detection"""
        lanes = traci.lane.getIDList()
        if lanes:
            has_emergency = self.sumo_utils.detect_emergency_vehicle(lanes[:5])
            self.assertIsInstance(has_emergency, bool)
            print(f"Emergency vehicle detected: {has_emergency}")
    
    def test_simulation_steps(self):
        """Test running multiple simulation steps"""
        for step in range(10):
            traci.simulationStep()
            
            # Call helper functions each step
            lanes = traci.lane.getIDList()
            if lanes:
                lane_id = lanes[0]
                queue = self.sumo_utils.get_queue_length(lane_id)
                speed = self.sumo_utils.get_avg_speed(lane_id)
                co2 = self.sumo_utils.get_co2_emission(lane_id)
                
                if step == 9:  # Last step
                    print(f"\nRan 10 simulation steps")
                    print(f"   Final step metrics:")
                    print(f"   - Queue: {queue}")
                    print(f"   - Speed: {speed:.2f}")
                    print(f"   - CO2: {co2:.2f}")
    
    def test_traffic_light_control(self):
        """Test traffic light control functions"""
        tl_ids = self.sumo_utils.get_all_intersections()
        
        if tl_ids:
            tl_id = tl_ids[0]
            
            # Get current phase
            current_phase = traci.trafficlight.getPhase(tl_id)
            print(f"Current phase for {tl_id}: {current_phase}")
            
            # Try setting phase
            try:
                traci.trafficlight.setPhase(tl_id, 0)
                new_phase = traci.trafficlight.getPhase(tl_id)
                self.assertEqual(new_phase, 0)
                print(f"Successfully changed phase to 0")
            except Exception as e:
                print(f"Could not change phase: {e}")

def run_manual_test():
    """Run manual test with detailed output"""
    print("\n" + "="*60)
    print("MANUAL SUMO TEST")
    print("="*60)
    
    if not os.path.exists(paths_config.SUMO_CONFIG):
        print(f"SUMO config not found at {paths_config.SUMO_CONFIG}")
        return False
    
    utils = SUMOUtils()
    
    try:
        # Start SUMO
        print("\nStarting SUMO...")
        traci.start(["sumo", "-c", paths_config.SUMO_CONFIG, "--no-warnings", "true"])
        
        for _ in range(5):
            traci.simulationStep()
        
        print("SUMO started successfully")
        
        # Run test steps
        print("\nRunning 10 test steps...")
        print("-" * 50)
        print(f"{'Step':<6} {'Time':<8} {'Vehicles':<10} {'Queue':<8} {'CO2':<10}")
        print("-" * 50)
        
        for step in range(10):
            traci.simulationStep()
            time = traci.simulation.getTime()
            vehicles = traci.vehicle.getIDCount()
            
            lanes = traci.lane.getIDList()
            if lanes:
                lane_id = lanes[0]
                queue = utils.get_queue_length(lane_id)
                co2 = utils.get_co2_emission(lane_id)
            else:
                queue = 0
                co2 = 0
            
            print(f"{step:<6} {time:<8.1f} {vehicles:<10} {queue:<8} {co2:<10.1f}")
        
        traci.close()
        print("\nManual test completed successfully")
        return True
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run manual test first
    success = run_manual_test()
    
    if success:
        # Run unit tests
        print("\nRunning unit tests...\n")
        unittest.main(argv=[''], verbosity=2, exit=False)
    else:
        sys.exit(1)
