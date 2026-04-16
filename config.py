"""
Central configuration file for Smart Traffic RL project
Contains all hyperparameters and constants
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class SUMOConfig:
    """SUMO simulation configuration"""
    # Network configuration
    GRID_SIZE: int = 3
    NUM_INTERSECTIONS: int = 9
    INTERSECTION_IDS: List[str] = None
    
    # Simulation parameters
    SIMULATION_STEPS: int = 3600
    STEP_LENGTH: float = 1.0  # seconds
    GUI: bool = False
    SUMO_CONFIG_PATH: str = "simulation/networks/grid/grid.sumocfg"
    
    # Traffic parameters
    MAX_VEHICLES: int = 500
    RUSH_HOUR_START: int = 100
    RUSH_HOUR_END: int = 300
    EMERGENCY_VEHICLES: int = 2
    
    # Lane configuration
    NUM_LANES: int = 2
    LANE_SPEED_LIMIT: float = 13.89  # 50 km/h in m/s
    
    def __post_init__(self):
        if self.INTERSECTION_IDS is None:
            self.INTERSECTION_IDS = [
                "n00", "n01", "n02",
                "n10", "n11", "n12",
                "n20", "n21", "n22"
            ]

@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    # Observation space
    OBSERVATION_DIM: int = 18
    MAX_QUEUE_LENGTH: int = 100
    MAX_SPEED: float = 30.0
    MAX_CO2: float = 10000.0
    MAX_PHASE_DURATION: int = 300
    
    # Action space
    NUM_ACTIONS: int = 3
    ACTION_KEEP_PHASE: int = 0
    ACTION_SWITCH_PHASE: int = 1
    ACTION_EMERGENCY: int = 2
    
    # Reward weights
    REWARD_THROUGHPUT_WEIGHT: float = 0.4
    REWARD_WAIT_WEIGHT: float = -0.4
    REWARD_EMISSIONS_WEIGHT: float = -0.2
    REWARD_EMERGENCY_BONUS: float = 100.0
    REWARD_EMERGENCY_PENALTY: float = -200.0
    
    # Multi-agent specific
    LOCAL_REWARD_WEIGHT: float = 0.6
    GLOBAL_REWARD_WEIGHT: float = 0.4
    
    # Emergency handling
    EMERGENCY_TIMEOUT_STEPS: int = 30
    EMERGENCY_CLEARANCE_BONUS: float = 100.0
    EMERGENCY_DELAY_PENALTY: float = -200.0

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Single agent curriculum
    STAGE_1_TIMESTEPS: int = 50000
    STAGE_2_TIMESTEPS: int = 100000
    STAGE_3_TIMESTEPS: int = 150000
    
    STAGE_1_MAX_VEHICLES: int = 100
    STAGE_2_MAX_VEHICLES: int = 300
    STAGE_3_MAX_VEHICLES: int = 500
    
    # Multi-agent
    MAPPO_TOTAL_TIMESTEPS: int = 300000
    
    # PPO hyperparameters
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    
    # Policy network architecture
    POLICY_HIDDEN_LAYERS: List[int] = None
    VALUE_HIDDEN_LAYERS: List[int] = None
    
    # Evaluation
    N_EVAL_EPISODES: int = 10
    EVAL_FREQUENCY: int = 5000
    
    # Logging
    TENSORBOARD_LOG_DIR: str = "./tensorboard_logs/"
    MODEL_SAVE_DIR: str = "./agents/"
    RESULTS_DIR: str = "./training/results/"
    
    def __post_init__(self):
        if self.POLICY_HIDDEN_LAYERS is None:
            self.POLICY_HIDDEN_LAYERS = [256, 256]
        if self.VALUE_HIDDEN_LAYERS is None:
            self.VALUE_HIDDEN_LAYERS = [256, 256]

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    REFRESH_RATE: int = 1  # seconds
    MAX_HISTORY_POINTS: int = 1000
    DEFAULT_PORT: int = 8501
    DEFAULT_HOST: str = "localhost"
    
    # Visualization colors
    COLOR_GREEN: str = "#00ff00"
    COLOR_YELLOW: str = "#ffff00"
    COLOR_RED: str = "#ff0000"
    
    # Phase colors mapping
    PHASE_COLORS: Dict[int, str] = None
    PHASE_LABELS: Dict[int, str] = None
    
    def __post_init__(self):
        if self.PHASE_COLORS is None:
            self.PHASE_COLORS = {
                0: "#00ff00",  # Green - NS green
                1: "#ffff00",  # Yellow - NS yellow
                2: "#ff0000",  # Red - EW green
                3: "#ffff00"   # Yellow - EW yellow
            }
        if self.PHASE_LABELS is None:
            self.PHASE_LABELS = {
                0: 'NS Green',
                1: 'NS Yellow',
                2: 'EW Green',
                3: 'EW Yellow'
            }

@dataclass
class PathsConfig:
    """File paths configuration"""
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    
    # SUMO files (relative sub-paths, absolute versions resolved in __post_init__)
    SIMULATION_DIR: str = "simulation"
    NETWORK_FILE: str = "simulation/networks/grid/grid_network.net.xml"
    ROUTES_FILE: str = "simulation/networks/grid/routes.rou.xml"
    SUMO_CONFIG: str = "simulation/networks/grid/grid.sumocfg"
    
    # Model files
    AGENTS_DIR: str = "agents"
    SINGLE_AGENT_MODEL: str = "agents/ppo_final.zip"
    SINGLE_AGENT_STAGE1: str = "agents/ppo_stage1.zip"
    SINGLE_AGENT_STAGE2: str = "agents/ppo_stage2.zip"
    MULTI_AGENT_MODEL: str = "agents/mappo_final.zip"
    
    # Results files
    RESULTS_DIR: str = "training/results"
    MULTI_RESULTS_DIR: str = "training/results_multi"
    FINAL_REPORT: str = "training/results/final_report.md"
    
    # Logs
    LOG_DIR: str = "logs"
    TENSORBOARD_LOG_DIR: str = "tensorboard_logs"
    TENSORBOARD_MULTI_LOG_DIR: str = "tensorboard_logs_multi"
    
    def __post_init__(self):
        """Resolve relative paths to absolute and create directories if they don't exist"""
        # Make all file/dir paths absolute based on PROJECT_ROOT
        self.SUMO_CONFIG  = os.path.join(self.PROJECT_ROOT, self.SUMO_CONFIG)
        self.NETWORK_FILE = os.path.join(self.PROJECT_ROOT, self.NETWORK_FILE)
        self.ROUTES_FILE  = os.path.join(self.PROJECT_ROOT, self.ROUTES_FILE)
        self.SIMULATION_DIR            = os.path.join(self.PROJECT_ROOT, self.SIMULATION_DIR)
        self.AGENTS_DIR                = os.path.join(self.PROJECT_ROOT, self.AGENTS_DIR)
        self.RESULTS_DIR               = os.path.join(self.PROJECT_ROOT, self.RESULTS_DIR)
        self.MULTI_RESULTS_DIR         = os.path.join(self.PROJECT_ROOT, self.MULTI_RESULTS_DIR)
        self.LOG_DIR                   = os.path.join(self.PROJECT_ROOT, self.LOG_DIR)
        self.TENSORBOARD_LOG_DIR       = os.path.join(self.PROJECT_ROOT, self.TENSORBOARD_LOG_DIR)
        self.TENSORBOARD_MULTI_LOG_DIR = os.path.join(self.PROJECT_ROOT, self.TENSORBOARD_MULTI_LOG_DIR)

        # Create directories if they don't exist
        for dir_path in [
            self.SIMULATION_DIR,
            self.AGENTS_DIR,
            self.RESULTS_DIR,
            self.MULTI_RESULTS_DIR,
            self.LOG_DIR,
            self.TENSORBOARD_LOG_DIR,
            self.TENSORBOARD_MULTI_LOG_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)

# Create singleton instances
sumo_config = SUMOConfig()
env_config = EnvironmentConfig()
training_config = TrainingConfig()
dashboard_config = DashboardConfig()
paths_config = PathsConfig()

# Export all configs
__all__ = [
    'sumo_config',
    'env_config', 
    'training_config',
    'dashboard_config',
    'paths_config'
]
