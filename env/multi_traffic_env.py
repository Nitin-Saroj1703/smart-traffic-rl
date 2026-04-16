"""
Multi-agent traffic environment for 3x3 grid of intersections
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import traci
import sumolib
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sumo_utils import SUMOUtils
from env.traffic_env import TrafficEnv
from config import paths_config, sumo_config as sc, env_config

class MultiTrafficEnv(ParallelEnv):
    """
    Multi-agent environment for 9 intersections in 3x3 grid
    
    Agents: 9 agents (intersection_0 to intersection_8)
    Observation space: Same as single TrafficEnv (17 dimensions + any multi-agent specifics)
    Action space: Same as single TrafficEnv (3 actions)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'name': 'multi_traffic_v1'}
    
    def __init__(
        self,
        sumo_config_file: Optional[str] = None,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        gui: Optional[bool] = None,
        emergency_timeout: Optional[int] = None,
        local_reward_weight: Optional[float] = None,
        global_reward_weight: Optional[float] = None
    ):
        super().__init__()
        
        self.sumo_config = sumo_config_file if sumo_config_file is not None else paths_config.SUMO_CONFIG
        self.max_steps = max_steps if max_steps is not None else sc.SIMULATION_STEPS
        self.render_mode = render_mode
        self.gui = gui if gui is not None else sc.GUI
        self.emergency_timeout = emergency_timeout if emergency_timeout is not None else env_config.EMERGENCY_TIMEOUT_STEPS
        self.local_reward_weight = local_reward_weight if local_reward_weight is not None else env_config.LOCAL_REWARD_WEIGHT
        self.global_reward_weight = global_reward_weight if global_reward_weight is not None else env_config.GLOBAL_REWARD_WEIGHT
        
        # Define all 9 intersection IDs (3x3 grid)
        self.intersection_ids = sc.INTERSECTION_IDS
        
        # Agent names
        self.possible_agents = [f"intersection_{i}" for i in range(sc.NUM_INTERSECTIONS)]
        self.agents = self.possible_agents[:]
        
        # Map agent names to intersection IDs
        self.agent_to_intersection = {
            f"intersection_{i}": self.intersection_ids[i] for i in range(sc.NUM_INTERSECTIONS)
        }
        
        # Initialize SUMO utilities
        self.sumo_utils = SUMOUtils()
        
        # Observation space (17 dimensions to match TrafficEnv)
        self._obs_space = spaces.Box(
            low=np.array([
                0, 0, 0, 0,  # Queue lengths
                0, 0, 0, 0,  # Average speeds
                0, 0, 0, 0,  # CO2 emissions
                0,            # Current phase
                0,            # Phase timer
                -1, -1,       # Time of day sin/cos
                0             # Emergency flag
            ], dtype=np.float32),
            high=np.array([
                env_config.MAX_QUEUE_LENGTH] * 4 +
                [env_config.MAX_SPEED] * 4 +
                [env_config.MAX_CO2] * 4 +
                [3,
                env_config.MAX_PHASE_DURATION,
                1, 1,
                1
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space (same for all agents)
        self._act_space = spaces.Discrete(3)
        
        # Define spaces for all agents
        self.observation_spaces = {agent: self._obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._act_space for agent in self.possible_agents}
        
        # State variables
        self.current_step = 0
        self.traci_connected = False
        self.agents_initialized = False
        
        # Per-agent state tracking
        self.agent_states = {
            agent: {
                'emergency_present': False,
                'emergency_start_step': None,
                'emergency_cleared': False,
                'last_phase': 0,
                'phase_start_time': 0,
                'vehicles_exited': 0,
                'cumulative_reward': 0,
                'lane_mapping': {}
            } for agent in self.possible_agents
        }
        
        # Adjacency matrix for 3x3 grid (for coordination analysis)
        self.adjacency = self._create_adjacency_matrix()
        
    def _create_adjacency_matrix(self) -> Dict[str, List[str]]:
        """Create adjacency mapping for 3x3 grid"""
        adj = {}
        grid_positions = {
            "n00": (0, 0), "n01": (0, 1), "n02": (0, 2),
            "n10": (1, 0), "n11": (1, 1), "n12": (1, 2),
            "n20": (2, 0), "n21": (2, 1), "n22": (2, 2)
        }
        
        for intersection, (row, col) in grid_positions.items():
            neighbors = []
            # Check all 4 directions
            if row > 0:
                neighbors.append(f"n{row-1}{col}")
            if row < 2:
                neighbors.append(f"n{row+1}{col}")
            if col > 0:
                neighbors.append(f"n{row}{col-1}")
            if col < 2:
                neighbors.append(f"n{row}{col+1}")
            
            agent_name = f"intersection_{self.intersection_ids.index(intersection)}"
            adj[agent_name] = [f"intersection_{self.intersection_ids.index(n)}" 
                              for n in neighbors]
        
        return adj
    
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
                "--no-warnings", "true",
                "--waiting-time-memory", "100"
            ]
            
            traci.start(sumo_cmd)
            self.traci_connected = True
            
            # Initialize simulation
            for _ in range(5):
                traci.simulationStep()
            
            # Initialize lane mappings for all agents
            for agent in self.agents:
                self._initialize_lane_mapping(agent)
    
    def _initialize_lane_mapping(self, agent: str):
        """Map incoming lanes to directions for an agent"""
        intersection_id = self.agent_to_intersection[agent]
        incoming_lanes = self.sumo_utils.get_incoming_lanes(intersection_id)
        
        lane_mapping = {}
        for lane_id in incoming_lanes:
            shape = traci.lane.getShape(lane_id)
            if len(shape) >= 2:
                start_x, start_y = shape[0][0], shape[0][1]
                end_x, end_y = shape[-1][0], shape[-1][1]
                
                dx = end_x - start_x
                dy = end_y - start_y
                
                if abs(dx) > abs(dy):
                    direction = 'W' if dx > 0 else 'E'
                else:
                    direction = 'N' if dy > 0 else 'S'
                
                lane_mapping[lane_id] = direction
        
        self.agent_states[agent]['lane_mapping'] = lane_mapping
    
    def _get_agent_observation(self, agent: str) -> np.ndarray:
        """Get observation for a specific agent"""
        intersection_id = self.agent_to_intersection[agent]
        lane_mapping = self.agent_states[agent]['lane_mapping']
        
        obs = []
        
        # Group lanes by direction
        direction_lanes = {'N': [], 'S': [], 'E': [], 'W': []}
        for lane_id, direction in lane_mapping.items():
            if direction in direction_lanes:
                direction_lanes[direction].append(lane_id)
        
        # Get metrics for each direction
        queue_lengths = []
        avg_speeds = []
        co2_emissions = []
        
        for direction in ['N', 'S', 'E', 'W']:
            lanes = direction_lanes[direction]
            
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
        
        # Current phase
        try:
            current_phase = traci.trafficlight.getPhase(intersection_id)
        except:
            current_phase = 0
        obs.append(current_phase)
        
        # Phase timer
        try:
            phase_duration = traci.trafficlight.getPhaseDuration(intersection_id)
        except:
            phase_duration = 0
        obs.append(min(phase_duration, env_config.MAX_PHASE_DURATION))
        
        # Time of day
        current_time = traci.simulation.getTime()
        time_of_day = (current_time % 3600) / 3600
        obs.append(math.sin(2 * math.pi * time_of_day))
        obs.append(math.cos(2 * math.pi * time_of_day))
        
        # Emergency vehicle flag
        incoming_lanes = list(lane_mapping.keys())
        emergency_present = self.sumo_utils.detect_emergency_vehicle(incoming_lanes)
        obs.append(1.0 if emergency_present else 0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _compute_local_reward(self, agent: str) -> float:
        """Compute local reward for a single agent"""
        intersection_id = self.agent_to_intersection[agent]
        incoming_lanes = list(self.agent_states[agent]['lane_mapping'].keys())
        
        # Throughput
        vehicles_exited = traci.simulation.getDepartedNumber() - self.agent_states[agent]['vehicles_exited']
        r_throughput = vehicles_exited
        self.agent_states[agent]['vehicles_exited'] = traci.simulation.getDepartedNumber()
        
        # Wait time (squared queue length)
        total_queue = 0
        for lane_id in incoming_lanes:
            queue = self.sumo_utils.get_queue_length(lane_id)
            total_queue += queue ** 2
        r_wait = total_queue / 100.0
        
        # Emissions
        total_co2 = 0
        for lane_id in incoming_lanes:
            total_co2 += self.sumo_utils.get_co2_emission(lane_id)
        r_emissions = total_co2 / 1000.0
        
        # Emergency handling
        r_emergency = self._compute_emergency_reward(agent)
        
        # Composite local reward
        local_reward = (env_config.REWARD_THROUGHPUT_WEIGHT * r_throughput) + \
                       (env_config.REWARD_WAIT_WEIGHT * r_wait) + \
                       (env_config.REWARD_EMISSIONS_WEIGHT * r_emissions) + r_emergency
        
        return local_reward
    
    def _compute_emergency_reward(self, agent: str) -> float:
        """Compute emergency-related reward for an agent"""
        incoming_lanes = list(self.agent_states[agent]['lane_mapping'].keys())
        emergency_present = self.sumo_utils.detect_emergency_vehicle(incoming_lanes)
        
        state = self.agent_states[agent]
        reward = 0
        
        if emergency_present and not state['emergency_present']:
            state['emergency_present'] = True
            state['emergency_start_step'] = self.current_step
            state['emergency_cleared'] = False
            
        elif not emergency_present and state['emergency_present']:
            state['emergency_present'] = False
            state['emergency_cleared'] = True
            
            if state['emergency_start_step'] is not None:
                steps_to_clear = self.current_step - state['emergency_start_step']
                
                if steps_to_clear <= self.emergency_timeout:
                    reward = env_config.EMERGENCY_CLEARANCE_BONUS
                else:
                    reward = env_config.EMERGENCY_DELAY_PENALTY
        
        return reward
    
    def _compute_global_reward(self) -> float:
        """Compute global reward as average of all local rewards"""
        local_rewards = []
        for agent in self.agents:
            try:
                # Compute local reward without emergency component for fairness
                intersection_id = self.agent_to_intersection[agent]
                incoming_lanes = list(self.agent_states[agent]['lane_mapping'].keys())
                
                # Simplified reward calculation
                total_queue = sum(self.sumo_utils.get_queue_length(lane) for lane in incoming_lanes)
                total_co2 = sum(self.sumo_utils.get_co2_emission(lane) for lane in incoming_lanes)
                
                reward = - (0.6 * total_queue + 0.4 * total_co2 / 100)
                local_rewards.append(reward)
            except:
                local_rewards.append(0)
        
        return np.mean(local_rewards) if local_rewards else 0
    
    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict]
    ]:
        """Execute actions for all agents"""
        
        # Apply actions
        for agent, action in actions.items():
            intersection_id = self.agent_to_intersection[agent]
            
            if action == 1:  # Switch phase
                try:
                    current_phase = traci.trafficlight.getPhase(intersection_id)
                    next_phase = (current_phase + 1) % 4
                    traci.trafficlight.setPhase(intersection_id, next_phase)
                    self.agent_states[agent]['last_phase'] = next_phase
                except:
                    pass
            elif action == 2:  # Emergency override
                try:
                    traci.trafficlight.setPhase(intersection_id, 0)
                except:
                    pass
        
        # Advance simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Compute global reward once
        global_reward = self._compute_global_reward()
        
        # Get observations and compute rewards for all agents
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent in self.agents:
            # Get observation
            observations[agent] = self._get_agent_observation(agent)
            
            # Compute rewards
            local_reward = self._compute_local_reward(agent)
            combined_reward = (self.local_reward_weight * local_reward + 
                             self.global_reward_weight * global_reward)
            rewards[agent] = combined_reward
            
            # Update cumulative reward
            self.agent_states[agent]['cumulative_reward'] += combined_reward
            
            # Check termination
            terminated = False
            truncated = self.current_step >= self.max_steps
            
            terminations[agent] = terminated
            truncations[agent] = truncated
            
            # Additional info
            infos[agent] = {
                'step': self.current_step,
                'local_reward': local_reward,
                'global_reward': global_reward,
                'cumulative_reward': self.agent_states[agent]['cumulative_reward'],
                'emergency_present': self.agent_states[agent]['emergency_present']
            }
        
        # Check if all agents are done
        if all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Dict]
    ]:
        """Reset the environment"""
        
        # Close existing connection
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        
        # Start new SUMO instance
        self._start_sumo()
        
        # Reset state
        self.current_step = 0
        self.agents = self.possible_agents[:]
        
        # Reset agent states
        for agent in self.agents:
            self.agent_states[agent].update({
                'emergency_present': False,
                'emergency_start_step': None,
                'emergency_cleared': False,
                'last_phase': 0,
                'phase_start_time': 0,
                'vehicles_exited': 0,
                'cumulative_reward': 0
            })
        
        # Get initial observations
        observations = {agent: self._get_agent_observation(agent) 
                       for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def render(self):
        """Render the environment"""
        pass
    
    def close(self):
        """Clean up resources"""
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
    
    def get_adjacent_intersections(self, agent: str) -> List[str]:
        """Get list of adjacent agents for coordination analysis"""
        return self.adjacency.get(agent, [])
