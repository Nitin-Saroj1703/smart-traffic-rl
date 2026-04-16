"""
Custom Gymnasium environment for Smart Traffic Signal Control
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
import time
from typing import Dict, List, Tuple, Optional

class TrafficSignalEnv(gym.Env):
    """
    Custom Environment for Traffic Signal Control using SUMO
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, 
                 sumo_config: str,
                 max_steps: int = 720,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.sumo_config = sumo_config
        self.max_steps = max_steps  # 720 steps * 5 sim-seconds = 3600 sim-seconds
        self.render_mode = render_mode
        self.current_step = 0
        
        # Define action space (traffic light phases)
        self.action_space = spaces.Discrete(4)  # 4 phases: NS-green, NS-yellow, EW-green, EW-yellow
        
        # Define observation space
        # [queue_length_north, queue_length_south, queue_length_east, queue_length_west,
        #  current_phase, elapsed_time]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(6,), dtype=np.float32
        )
        
        # SUMO connection
        self.traci_connected = False
        self._sumo_label = "default"
        
    def _start_sumo(self):
        """Initialize SUMO simulation"""
        if not self.traci_connected:
            sumo_binary = "sumo"  # or "sumo-gui" for GUI version
            sumo_cmd = [sumo_binary, "-c", self.sumo_config,
                        "--no-step-log", "true",
                        "--waiting-time-memory", "1000",
                        "--no-warnings", "true"]
            
            if self.render_mode == "human":
                sumo_cmd = ["sumo-gui", "-c", self.sumo_config, "--start",
                            "--waiting-time-memory", "1000"]
            
            # Use a unique label for each connection to avoid conflicts
            self._sumo_label = f"sim_{id(self)}_{time.time()}"
            traci.start(sumo_cmd, label=self._sumo_label)
            self._conn = traci.getConnection(self._sumo_label)
            self.traci_connected = True
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from SUMO"""
        # Get queue lengths (simplified - using halted vehicles count)
        lanes = ["north_in_0", "south_in_0", "east_in_0", "west_in_0"]
        queues = []
        
        for lane_id in lanes:
            try:
                # Count halted vehicles
                vehicles = self._conn.lane.getLastStepVehicleIDs(lane_id)
                halted = sum(1 for v in vehicles if self._conn.vehicle.getSpeed(v) < 0.1)
                queues.append(min(halted, 100))  # Clip to obs space max
            except Exception:
                queues.append(0)
        
        # Get current phase
        tl_id = "center"
        try:
            current_phase = float(self._conn.trafficlight.getPhase(tl_id))
            elapsed_time = float(self._conn.trafficlight.getPhaseDuration(tl_id))
        except Exception:
            current_phase = 0.0
            elapsed_time = 0.0
        
        obs = np.array(queues + [current_phase, elapsed_time], dtype=np.float32)
        return obs
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on:
        - Minimize queue lengths
        - Minimize CO2 emissions
        - Minimize wait times
        """
        try:
            # Get queue lengths
            queues = self._get_observation()[:4]
            queue_penalty = -np.sum(queues) * 0.1
            
            # Get CO2 emissions (simplified)
            total_co2 = 0
            for veh_id in self._conn.vehicle.getIDList():
                total_co2 += self._conn.vehicle.getCO2Emission(veh_id)
            co2_penalty = -total_co2 * 0.001
            
            # Get wait times
            total_wait = 0
            for veh_id in self._conn.vehicle.getIDList():
                total_wait += self._conn.vehicle.getWaitingTime(veh_id)
            wait_penalty = -total_wait * 0.01
            
            reward = queue_penalty + co2_penalty + wait_penalty
        except Exception:
            reward = 0.0
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        try:
            # Set traffic light phase
            tl_id = "center"
            self._conn.trafficlight.setPhase(tl_id, action)
            
            # Simulate for duration
            for _ in range(5):  # 5 seconds per step
                self._conn.simulationStep()
            
            self.current_step += 1
            
            # Check if SUMO simulation ended (all vehicles departed and arrived)
            sim_time = self._conn.simulation.getTime()
            
            # Get observation
            obs = self._get_observation()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            terminated = self.current_step >= self.max_steps or sim_time >= 3600
            truncated = False
            
            # Additional info
            info = {
                "step": self.current_step,
                "sim_time": sim_time,
                "total_waiting_time": sum(self._conn.vehicle.getWaitingTime(v) 
                                         for v in self._conn.vehicle.getIDList()),
                "total_co2": sum(self._conn.vehicle.getCO2Emission(v) 
                               for v in self._conn.vehicle.getIDList())
            }
        except Exception as e:
            # SUMO connection dropped — end the episode gracefully
            obs = np.zeros(6, dtype=np.float32)
            reward = 0.0
            terminated = True
            truncated = False
            info = {"step": self.current_step, "error": str(e)}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Safely close existing connection
        if self.traci_connected:
            try:
                self._conn.close()
            except Exception:
                pass
            self.traci_connected = False
        
        self._start_sumo()
        self.current_step = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.traci_connected:
            # SUMO-GUI handles rendering automatically
            pass
    
    def close(self):
        """Clean up resources"""
        if self.traci_connected:
            try:
                self._conn.close()
            except Exception:
                pass
            self.traci_connected = False
