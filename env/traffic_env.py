"""
Custom Gymnasium environment for traffic signal control at a single intersection
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sumo_utils import SUMOUtils
from config import paths_config, sumo_config as sc, env_config

class TrafficEnv(gym.Env):
    """
    Custom Environment for single intersection traffic signal control
    
    Observation Space (18 dimensions):
        - Queue lengths for N, S, E, W lanes (4 floats)
        - Average speeds for N, S, E, W lanes (4 floats)
        - CO2 emissions for N, S, E, W lanes (4 floats)
        - Current signal phase (1 integer 0-3)
        - Phase timer (1 float)
        - Time of day encoded as sin/cos (2 floats)
        - Emergency vehicle flag (1 binary)
    
    Action Space:
        - 0: Keep current phase
        - 1: Switch to next phase
        - 2: Emergency override
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(
        self,
        sumo_config_file: Optional[str] = None,
        intersection_id: str = "n11",  # Center intersection
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        gui: Optional[bool] = None,
        emergency_timeout: Optional[int] = None
    ):
        super().__init__()
        
        self.sumo_config = sumo_config_file if sumo_config_file is not None else paths_config.SUMO_CONFIG
        self.intersection_id = intersection_id
        self.max_steps = max_steps if max_steps is not None else sc.SIMULATION_STEPS
        self.render_mode = render_mode
        self.gui = gui if gui is not None else sc.GUI
        self.emergency_timeout = emergency_timeout if emergency_timeout is not None else env_config.EMERGENCY_TIMEOUT_STEPS
        
        # Initialize SUMO utilities
        self.sumo_utils = SUMOUtils()
        
        # Define action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: 18-dimensional Box
        self.observation_space = spaces.Box(
            low=np.array([
                0, 0, 0, 0,  # Queue lengths (min 0)
                0, 0, 0, 0,  # Average speeds (min 0)
                0, 0, 0, 0,  # CO2 emissions (min 0)
                0,            # Current phase (0-3)
                0,            # Phase timer (min 0)
                -1, -1,       # Time of day sin/cos (-1 to 1)
                0             # Emergency flag (0 or 1)
            ], dtype=np.float32),
            high=np.array([
                env_config.MAX_QUEUE_LENGTH] * 4 +  # Queue lengths
                [env_config.MAX_SPEED] * 4 +      # Average speeds
                [env_config.MAX_CO2] * 4 +  # CO2 emissions
                [3,                    # Current phase (max 3)
                env_config.MAX_PHASE_DURATION,                  # Phase timer
                1, 1,                 # Time of day sin/cos (-1 to 1)
                1                     # Emergency flag (0 or 1)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.traci_connected = False
        self.emergency_vehicle_present = False
        self.emergency_start_step = None
        self.emergency_cleared = False
        self.emergency_cleared_step = None
        self.last_phase = 0
        self.phase_start_time = 0
        self.cumulative_wait_time = 0
        self.vehicles_exited = 0
        self.episode_reward = 0
        
        # Lane mapping for the intersection
        self.incoming_lanes = []
        self.lane_directions = {}  # Maps lane_id to direction (N, S, E, W)
        
    def _start_sumo(self):
        """Initialize SUMO simulation"""
        if not self.traci_connected:
            sumo_binary = "sumo-gui" if self.gui else "sumo"
            
            # Ensure SUMO_HOME is set and bin is in path for traci to find the binary
            sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
            if sumo_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")
            
            sumo_cmd = [
                sumo_binary,
                "-c", self.sumo_config,
                "--start",
                "--time-to-teleport", "-1",
                "--no-warnings", "true"
            ]
            
            if self.render_mode == "human" and not self.gui:
                sumo_cmd.append("--gui-settings-file")
                sumo_cmd.append("simulation/gui-settings.xml")
            
            traci.start(sumo_cmd)
            self.traci_connected = True
            
            # Wait for simulation to initialize
            for _ in range(5):
                traci.simulationStep()
            
            # Initialize lane mapping
            self._initialize_lane_mapping()
    
    def _initialize_lane_mapping(self):
        """Map incoming lanes to directions (N, S, E, W)"""
        self.incoming_lanes = self.sumo_utils.get_incoming_lanes(self.intersection_id)
        
        # Determine direction based on lane geometry
        for lane_id in self.incoming_lanes:
            shape = traci.lane.getShape(lane_id)
            if len(shape) >= 2:
                start_x, start_y = shape[0][0], shape[0][1]
                end_x, end_y = shape[-1][0], shape[-1][1]
                
                dx = end_x - start_x
                dy = end_y - start_y
                
                if abs(dx) > abs(dy):
                    if dx > 0:
                        self.lane_directions[lane_id] = 'W'  # Coming from west
                    else:
                        self.lane_directions[lane_id] = 'E'  # Coming from east
                else:
                    if dy > 0:
                        self.lane_directions[lane_id] = 'N'  # Coming from north
                    else:
                        self.lane_directions[lane_id] = 'S'  # Coming from south
    
    def _get_observation(self) -> np.ndarray:
        """
        Build the 18-dimensional observation vector
        
        Returns:
            numpy array of shape (18,)
        """
        obs = []
        
        # Group lanes by direction
        direction_lanes = {'N': [], 'S': [], 'E': [], 'W': []}
        for lane_id, direction in self.lane_directions.items():
            if direction in direction_lanes:
                direction_lanes[direction].append(lane_id)
        
        # Get metrics for each direction
        queue_lengths = []
        avg_speeds = []
        co2_emissions = []
        
        for direction in ['N', 'S', 'E', 'W']:
            lanes = direction_lanes[direction]
            
            # Queue length (sum across all lanes in this direction)
            queue = 0
            speed_sum = 0
            speed_count = 0
            co2 = 0
            
            for lane_id in lanes:
                queue += self.sumo_utils.get_queue_length(lane_id)
                speed = self.sumo_utils.get_avg_speed(lane_id)
                if speed > 0:
                    speed_sum += speed
                    speed_count += 1
                co2 += self.sumo_utils.get_co2_emission(lane_id)
            
            queue_lengths.append(min(queue, env_config.MAX_QUEUE_LENGTH))
            
            avg_speed = speed_sum / speed_count if speed_count > 0 else 0
            avg_speeds.append(min(avg_speed, env_config.MAX_SPEED))
            
            co2_emissions.append(min(co2, env_config.MAX_CO2))
        
        obs.extend(queue_lengths)
        obs.extend(avg_speeds)
        obs.extend(co2_emissions)
        
        # Current signal phase (0-3)
        try:
            current_phase = traci.trafficlight.getPhase(self.intersection_id)
        except:
            current_phase = 0
        obs.append(current_phase)
        
        # Phase timer (how long current phase has been active)
        try:
            # Note: getPhaseDuration returns the TOTAL defined duration of the current phase
            # But we might want elapsed time. However, following original code's logic.
            phase_duration = traci.trafficlight.getPhaseDuration(self.intersection_id)
        except:
            phase_duration = 0
        obs.append(min(phase_duration, env_config.MAX_PHASE_DURATION))
        
        # Time of day encoded as sin/cos
        current_time = traci.simulation.getTime()
        time_of_day = (current_time % 3600) / 3600  # Normalize to [0, 1] for 1-hour cycle
        obs.append(math.sin(2 * math.pi * time_of_day))
        obs.append(math.cos(2 * math.pi * time_of_day))
        
        # Emergency vehicle flag
        emergency_present = self._check_emergency_vehicle()
        obs.append(1.0 if emergency_present else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _check_emergency_vehicle(self) -> bool:
        """Check if emergency vehicle is present in any incoming lane"""
        return self.sumo_utils.detect_emergency_vehicle(self.incoming_lanes)
    
    def _compute_reward(self) -> float:
        """
        Compute composite reward:
        R = 0.4 * R_throughput - 0.4 * R_wait - 0.2 * R_emissions + R_emergency
        """
        # R_throughput: vehicles that exited the intersection this step
        vehicles_exited = traci.simulation.getDepartedNumber() - self.vehicles_exited
        r_throughput = vehicles_exited
        self.vehicles_exited = traci.simulation.getDepartedNumber()
        
        # R_wait: sum of squared queue lengths (normalized)
        total_queue = 0
        for lane_id in self.incoming_lanes:
            queue = self.sumo_utils.get_queue_length(lane_id)
            total_queue += queue ** 2  # Square to penalize long queues more heavily
        
        r_wait = total_queue / 100.0  # Normalize
        
        # R_emissions: total CO2 across all lanes (normalized)
        total_co2 = 0
        for lane_id in self.incoming_lanes:
            total_co2 += self.sumo_utils.get_co2_emission(lane_id)
        
        r_emissions = total_co2 / 1000.0  # Normalize to reasonable range
        
        # R_emergency: bonus/penalty for emergency vehicle handling
        r_emergency = 0
        emergency_present = self._check_emergency_vehicle()
        
        if emergency_present and not self.emergency_vehicle_present:
            # Emergency vehicle just appeared
            self.emergency_vehicle_present = True
            self.emergency_start_step = self.current_step
            self.emergency_cleared = False
            
        elif not emergency_present and self.emergency_vehicle_present:
            # Emergency vehicle just cleared
            self.emergency_vehicle_present = False
            self.emergency_cleared = True
            self.emergency_cleared_step = self.current_step
            
            # Calculate time to clear
            if self.emergency_start_step is not None:
                steps_to_clear = self.current_step - self.emergency_start_step
                
                if steps_to_clear <= self.emergency_timeout:
                    r_emergency = env_config.EMERGENCY_CLEARANCE_BONUS  # Bonus for quick clearance
                    print(f"✅ Emergency vehicle cleared in {steps_to_clear} steps! Bonus +{env_config.EMERGENCY_CLEARANCE_BONUS}")
                else:
                    r_emergency = env_config.EMERGENCY_DELAY_PENALTY  # Penalty for slow clearance
                    print(f"⚠️ Emergency vehicle took {steps_to_clear} steps! Penalty {env_config.EMERGENCY_DELAY_PENALTY}")
        
        # Composite reward
        reward = (env_config.REWARD_THROUGHPUT_WEIGHT * r_throughput) + \
                 (env_config.REWARD_WAIT_WEIGHT * r_wait) + \
                 (env_config.REWARD_EMISSIONS_WEIGHT * r_emissions) + r_emergency
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0=keep phase, 1=switch phase, 2=emergency override
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action
        if action == 1:  # Switch to next phase
            try:
                current_phase = traci.trafficlight.getPhase(self.intersection_id)
                num_phases = 4  # Assuming 4 phases
                next_phase = (current_phase + 1) % num_phases
                traci.trafficlight.setPhase(self.intersection_id, next_phase)
                self.last_phase = next_phase
                self.phase_start_time = self.current_step
            except:
                pass
                
        elif action == 2:  # Emergency override
            # Set all lanes red except for emergency vehicle path
            try:
                # Force a special phase that prioritizes emergency vehicle
                # In a real implementation, this would use a dedicated emergency phase
                traci.trafficlight.setPhase(self.intersection_id, 0)  # Simplified
                # Could implement more sophisticated emergency routing here
            except:
                pass
        
        # Action 0: Keep current phase (do nothing)
        
        # Advance simulation by 1 second (multiple steps for stability)
        for _ in range(1):
            traci.simulationStep()
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            'step': self.current_step,
            'total_queue': sum(self._get_observation()[:4]),
            'total_co2': sum(self._get_observation()[8:12]),
            'emergency_present': self.emergency_vehicle_present,
            'episode_reward': self.episode_reward
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Close existing connection if any
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        
        # Start new SUMO instance
        self._start_sumo()
        
        # Reset state variables
        self.current_step = 0
        self.emergency_vehicle_present = False
        self.emergency_start_step = None
        self.emergency_cleared = False
        self.emergency_cleared_step = None
        self.last_phase = 0
        self.phase_start_time = 0
        self.vehicles_exited = 0
        self.episode_reward = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and not self.gui:
            # SUMO-GUI handles rendering when gui=True
            pass
    
    def close(self):
        """Clean up resources"""
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
