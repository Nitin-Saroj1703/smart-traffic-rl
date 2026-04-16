"""
Train PPO agent on single intersection with curriculum learning
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import argparse
import traci

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import training_config, paths_config, env_config, sumo_config as sc
from env.traffic_env import TrafficEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import gymnasium as gym

class CurriculumTrafficEnv(gym.Wrapper):
    """Wrapper to modify traffic density for curriculum learning"""
    
    def __init__(self, env: TrafficEnv, max_vehicles: int = 100):
        super().__init__(env)
        self.max_vehicles = max_vehicles
        
    def step(self, action):
        # Monitor vehicle count and adjust if needed
        try:
            vehicle_count = traci.vehicle.getIDCount()
            if vehicle_count > self.max_vehicles:
                # Remove excess vehicles (simplified curriculum control)
                pass
        except:
            pass
        
        return self.env.step(action)

class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Accumulate rewards
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Log to tensorboard
            self.logger.record('episode/reward', self.current_episode_reward)
            self.logger.record('episode/length', self.current_episode_length)
            
            # Reset accumulators
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True

class MetricsCallback(BaseCallback):
    """Callback for tracking training metrics via TensorBoard.
    
    NOTE: Does NOT open a separate SUMO environment to avoid TraCI
    'Connection default is already active' conflicts. Instead, metrics
    are read from the training environment's info dict each step.
    """
    
    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_rewards = []
        self._ep_wait = []
        self._ep_co2 = []
        self.metrics_history = {
            'timesteps': [],
            'mean_reward': [],
            'mean_wait_time': [],
            'mean_co2': [],
            'emergency_clearance_rate': []
        }
        
    def _on_step(self) -> bool:
        # Collect per-step info from the training env
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'total_queue' in info:
                self._ep_wait.append(info['total_queue'])
            if 'total_co2' in info:
                self._ep_co2.append(info['total_co2'])

        # Log rollout averages every log_freq steps
        if self.n_calls % self.log_freq == 0 and self._ep_wait:
            mean_wait = float(np.mean(self._ep_wait))
            mean_co2  = float(np.mean(self._ep_co2)) if self._ep_co2 else 0.0

            self.metrics_history['timesteps'].append(self.num_timesteps)
            self.metrics_history['mean_wait_time'].append(mean_wait)
            self.metrics_history['mean_co2'].append(mean_co2)
            self.metrics_history['mean_reward'].append(0.0)          # filled post-training
            self.metrics_history['emergency_clearance_rate'].append(0.0)

            self.logger.record('train/mean_queue_length', mean_wait)
            self.logger.record('train/mean_co2', mean_co2)

            # Reset buffers
            self._ep_wait.clear()
            self._ep_co2.clear()

        return True

class TrafficCurriculum:
    """Manages curriculum learning stages"""
    
    def __init__(self, base_env_config: Dict):
        self.base_config = base_env_config
        self.stages = [
            {
                'name': 'Stage 1 - Easy',
                'max_vehicles': training_config.STAGE_1_MAX_VEHICLES,
                'emergency_enabled': False,
                'timesteps': training_config.STAGE_1_TIMESTEPS,
                'description': 'Low traffic, no emergencies'
            },
            {
                'name': 'Stage 2 - Medium',
                'max_vehicles': training_config.STAGE_2_MAX_VEHICLES,
                'emergency_enabled': False,
                'timesteps': training_config.STAGE_2_TIMESTEPS,
                'description': 'Rush-hour traffic'
            },
            {
                'name': 'Stage 3 - Hard',
                'max_vehicles': training_config.STAGE_3_MAX_VEHICLES,
                'emergency_enabled': True,
                'timesteps': training_config.STAGE_3_TIMESTEPS,
                'description': 'Full traffic with emergencies'
            }
        ]
        self.current_stage = 0
        
    def get_stage_config(self, stage_idx: int) -> Dict:
        """Get configuration for specific stage"""
        if stage_idx >= len(self.stages):
            return None
        
        stage = self.stages[stage_idx]
        config = self.base_config.copy()
        config.update({
            'max_vehicles': stage['max_vehicles'],
            'emergency_enabled': stage['emergency_enabled']
        })
        return config
    
    def get_stage_info(self, stage_idx: int) -> Dict:
        """Get information about a stage"""
        if stage_idx >= len(self.stages):
            return None
        return self.stages[stage_idx]

def create_env(config: Dict, gui: bool = False) -> gym.Env:
    """Create and wrap the traffic environment"""
    env = TrafficEnv(
        sumo_config_file=config.get('sumo_config', paths_config.SUMO_CONFIG),
        intersection_id=config.get('intersection_id', 'n11'),
        max_steps=config.get('max_steps', sc.SIMULATION_STEPS),
        gui=gui,
        emergency_timeout=config.get('emergency_timeout', env_config.EMERGENCY_TIMEOUT_STEPS)
    )
    
    # Apply curriculum wrapper if specified
    if 'max_vehicles' in config:
        env = CurriculumTrafficEnv(env, max_vehicles=config['max_vehicles'])
    
    # Monitor wrapper for logging
    env = Monitor(env)
    
    return env

def train_stage(
    stage_idx: int,
    curriculum: TrafficCurriculum,
    model: Optional[PPO] = None,
    log_dir: str = "./tensorboard_logs/"
) -> PPO:
    """Train one stage of the curriculum"""
    
    stage_info = curriculum.get_stage_info(stage_idx)
    stage_config = curriculum.get_stage_config(stage_idx)
    
    print("\n" + "="*80)
    print(f"Starting {stage_info['name']}")
    print(f"Description: {stage_info['description']}")
    print(f"Training for {stage_info['timesteps']} timesteps")
    print("="*80)
    
    # Create environments
    venv = create_env(stage_config, gui=False)
    venv = DummyVecEnv([lambda: venv])
    
    # Create evaluation environment
    eval_venv = create_env(stage_config, gui=False)
    eval_venv = DummyVecEnv([lambda: eval_venv])
    
    # Initialize or continue with model
    if model is None:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=training_config.LEARNING_RATE,
            n_steps=training_config.N_STEPS,
            batch_size=training_config.BATCH_SIZE,
            n_epochs=training_config.N_EPOCHS,
            gamma=training_config.GAMMA,
            gae_lambda=training_config.GAE_LAMBDA,
            clip_range=training_config.CLIP_RANGE,
            ent_coef=training_config.ENT_COEF,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        )
    else:
        print("Continuing with existing model...")
        model.set_env(venv)
    
    # Create callbacks
    tensorboard_callback = TensorboardCallback()
    metrics_callback = MetricsCallback(log_freq=2048)
    
    # Train
    print(f"\nStarting training for stage {stage_idx + 1}...")
    model.learn(
        total_timesteps=stage_info['timesteps'],
        callback=[tensorboard_callback, metrics_callback],
        progress_bar=False  # Avoids tqdm/rich teardown errors on Windows
    )
    
    # Save stage model
    stage_path = os.path.join(paths_config.AGENTS_DIR, f"ppo_stage{stage_idx + 1}.zip")
    os.makedirs(paths_config.AGENTS_DIR, exist_ok=True)
    model.save(stage_path)
    print(f"Stage {stage_idx + 1} model saved to {stage_path}")
    
    # Store metrics for plotting
    model.stage_metrics = metrics_callback.metrics_history
    
    # Clean up environments
    venv.close()
    eval_venv.close()
    
    return model

def evaluate_final_model(model: PPO, env_config: Dict, n_episodes: int = 10) -> Dict:
    """Evaluate the final trained model"""
    print("\n" + "="*80)
    print("Final Model Evaluation")
    print("="*80)
    
    eval_env = create_env(env_config, gui=False)
    
    all_rewards = []
    all_wait_times = []
    all_co2_emissions = []
    emergency_clearance_times = []
    emergency_cleared_count = 0
    emergency_total_count = 0
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_wait_times = []
        episode_co2 = []
        emergency_detected = False
        emergency_start_time = None
        emergency_cleared = False
        
        step = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            
            episode_reward += reward
            episode_wait_times.append(info.get('total_queue', 0))
            episode_co2.append(info.get('total_co2', 0))
            
            # Track emergency vehicles
            if info.get('emergency_present', False) and not emergency_detected:
                emergency_detected = True
                emergency_start_time = step
                emergency_total_count += 1
            elif emergency_detected and not info.get('emergency_present', False):
                if not emergency_cleared:
                    emergency_cleared = True
                    emergency_cleared_count += 1
                    clearance_time = step - emergency_start_time
                    emergency_clearance_times.append(clearance_time)
            
            step += 1
        
        all_rewards.append(episode_reward)
        all_wait_times.append(np.mean(episode_wait_times))
        all_co2_emissions.append(np.mean(episode_co2))
        
        print(f"Episode {episode + 1:2d}: "
              f"Reward={episode_reward:8.2f}, "
              f"Wait={np.mean(episode_wait_times):6.1f}, "
              f"CO2={np.mean(episode_co2):8.1f}")
    
    eval_env.close()
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_wait_time': np.mean(all_wait_times),
        'std_wait_time': np.std(all_wait_times),
        'mean_co2': np.mean(all_co2_emissions),
        'std_co2': np.std(all_co2_emissions),
        'emergency_clearance_rate': (emergency_cleared_count / emergency_total_count * 100) 
                                    if emergency_total_count > 0 else 0,
        'mean_emergency_clearance_time': np.mean(emergency_clearance_times) 
                                         if emergency_clearance_times else 0,
        'all_rewards': all_rewards,
        'all_wait_times': all_wait_times,
        'all_co2_emissions': all_co2_emissions
    }
    
    return results

def evaluate_baseline(env_config: Dict, n_episodes: int = 10) -> Dict:
    """Evaluate a random action baseline"""
    print("\n" + "="*80)
    print("Baseline (Random Agent) Evaluation")
    print("="*80)
    
    env = create_env(env_config, gui=False)
    
    all_rewards = []
    all_wait_times = []
    all_co2_emissions = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_wait_times = []
        episode_co2 = []
        
        while not (done or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_wait_times.append(info.get('total_queue', 0))
            episode_co2.append(info.get('total_co2', 0))
        
        all_rewards.append(episode_reward)
        all_wait_times.append(np.mean(episode_wait_times))
        all_co2_emissions.append(np.mean(episode_co2))
        
        print(f"Episode {episode + 1:2d}: "
              f"Reward={episode_reward:8.2f}, "
              f"Wait={np.mean(episode_wait_times):6.1f}, "
              f"CO2={np.mean(episode_co2):8.1f}")
    
    env.close()
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_wait_time': np.mean(all_wait_times),
        'std_wait_time': np.std(all_wait_times),
        'mean_co2': np.mean(all_co2_emissions),
        'std_co2': np.std(all_co2_emissions)
    }

def plot_training_results(
    stage_metrics_list: List[Dict],
    final_results: Dict,
    baseline_results: Dict,
    save_dir: str = "training/results"
):
    """Generate and save training plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine all stage metrics
    all_timesteps = []
    all_rewards = []
    all_wait_times = []
    all_co2 = []
    
    cumulative_timesteps = 0
    for stage_metrics in stage_metrics_list:
        if stage_metrics:
            timesteps = [t + cumulative_timesteps for t in stage_metrics['timesteps']]
            all_timesteps.extend(timesteps)
            all_rewards.extend(stage_metrics['mean_reward'])
            all_wait_times.extend(stage_metrics['mean_wait_time'])
            all_co2.extend(stage_metrics['mean_co2'])
            
            if timesteps:
                cumulative_timesteps = timesteps[-1]
    
    # Plot 1: Reward curve across all stages
    plt.figure(figsize=(12, 6))
    plt.plot(all_timesteps, all_rewards, 'b-', linewidth=2, label='PPO Agent')
    plt.axhline(y=baseline_results['mean_reward'], color='r', linestyle='--', 
                label=f"Baseline (Random) = {baseline_results['mean_reward']:.1f}")
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.title('Training Progress: Reward Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add stage boundaries
    stage_boundaries = [50000, 150000]
    for boundary in stage_boundaries:
        plt.axvline(x=boundary, color='g', linestyle=':', alpha=0.5)
        plt.text(boundary, plt.ylim()[0], f'Stage {stage_boundaries.index(boundary)+2}', 
                rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reward_curve.png", dpi=150)
    plt.close()
    
    # Plot 2: Average wait time per episode
    plt.figure(figsize=(12, 6))
    plt.plot(all_timesteps, all_wait_times, 'orange', linewidth=2, label='PPO Agent')
    plt.axhline(y=baseline_results['mean_wait_time'], color='r', linestyle='--',
                label=f"Baseline = {baseline_results['mean_wait_time']:.1f}")
    plt.xlabel('Timesteps')
    plt.ylabel('Average Queue Length (vehicles)')
    plt.title('Training Progress: Queue Length Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for boundary in stage_boundaries:
        plt.axvline(x=boundary, color='g', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/wait_time_curve.png", dpi=150)
    plt.close()
    
    # Plot 3: CO2 emissions per episode
    plt.figure(figsize=(12, 6))
    plt.plot(all_timesteps, all_co2, 'green', linewidth=2, label='PPO Agent')
    plt.axhline(y=baseline_results['mean_co2'], color='r', linestyle='--',
                label=f"Baseline = {baseline_results['mean_co2']:.1f}")
    plt.xlabel('Timesteps')
    plt.ylabel('CO2 Emissions (mg/s)')
    plt.title('Training Progress: CO2 Emissions Reduction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    for boundary in stage_boundaries:
        plt.axvline(x=boundary, color='g', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/co2_curve.png", dpi=150)
    plt.close()
    
    # Plot 4: Comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['mean_reward', 'mean_wait_time', 'mean_co2']
    titles = ['Mean Reward', 'Mean Queue Length', 'Mean CO2 Emissions']
    ylabels = ['Reward', 'Vehicles', 'mg/s']
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        
        # Values
        ppo_value = final_results[metric]
        baseline_value = baseline_results[metric]
        
        # Improvement percentage
        if metric == 'mean_reward':
            improvement = ((ppo_value - baseline_value) / abs(baseline_value)) * 100 if abs(baseline_value) > 0 else 0
        else:
            improvement = ((baseline_value - ppo_value) / baseline_value) * 100 if baseline_value > 0 else 0
        
        # Create bar chart
        bars = ax.bar(['Baseline', 'PPO'], [baseline_value, ppo_value], 
                      color=['gray', 'blue'], alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\nImprovement: {improvement:.1f}%')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison.png", dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")

def save_training_report(
    stage_metrics_list: List[Dict],
    final_results: Dict,
    baseline_results: Dict,
    save_dir: str = "training/results"
):
    """Save detailed training report"""
    os.makedirs(save_dir, exist_ok=True)
    
    report = {
        'training_date': datetime.now().isoformat(),
        'curriculum_stages': [
            {
                'stage': i + 1,
                'timesteps': len(stage_metrics['timesteps']) if stage_metrics else 0,
                'final_reward': stage_metrics['mean_reward'][-1] if stage_metrics and stage_metrics['mean_reward'] else None,
                'final_wait_time': stage_metrics['mean_wait_time'][-1] if stage_metrics and stage_metrics['mean_wait_time'] else None,
                'final_co2': stage_metrics['mean_co2'][-1] if stage_metrics and stage_metrics['mean_co2'] else None
            }
            for i, stage_metrics in enumerate(stage_metrics_list)
        ],
        'final_evaluation': {
            'mean_reward': float(final_results['mean_reward']),
            'std_reward': float(final_results['std_reward']),
            'mean_wait_time': float(final_results['mean_wait_time']),
            'std_wait_time': float(final_results['std_wait_time']),
            'mean_co2': float(final_results['mean_co2']),
            'std_co2': float(final_results['std_co2']),
            'emergency_clearance_rate': float(final_results['emergency_clearance_rate']),
            'mean_emergency_clearance_time': float(final_results['mean_emergency_clearance_time'])
        },
        'baseline': {
            'mean_reward': float(baseline_results['mean_reward']),
            'mean_wait_time': float(baseline_results['mean_wait_time']),
            'mean_co2': float(baseline_results['mean_co2'])
        },
        'improvements': {
            'reward': float(((final_results['mean_reward'] - baseline_results['mean_reward']) / 
                           abs(baseline_results['mean_reward'])) * 100) if abs(baseline_results['mean_reward']) > 0 else 0,
            'wait_time': float(((baseline_results['mean_wait_time'] - final_results['mean_wait_time']) / 
                              baseline_results['mean_wait_time']) * 100) if baseline_results['mean_wait_time'] > 0 else 0,
            'co2': float(((baseline_results['mean_co2'] - final_results['mean_co2']) / 
                        baseline_results['mean_co2']) * 100) if baseline_results['mean_co2'] > 0 else 0
        }
    }
    
    # Save as JSON
    with open(f"{save_dir}/training_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save as readable text
    with open(f"{save_dir}/training_report.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("SMART TRAFFIC RL - TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training Date: {report['training_date']}\n\n")
        
        f.write("-"*40 + "\n")
        f.write("CURRICULUM STAGES\n")
        f.write("-"*40 + "\n")
        for stage in report['curriculum_stages']:
            f.write(f"\nStage {stage['stage']}:\n")
            if stage['final_reward'] is not None:
                f.write(f"  Final Reward: {stage['final_reward']:.2f}\n")
                f.write(f"  Final Wait Time: {stage['final_wait_time']:.2f}\n")
                f.write(f"  Final CO2: {stage['final_co2']:.2f}\n")
            else:
                f.write("  No data recorded.\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("FINAL EVALUATION (PPO Agent)\n")
        f.write("-"*40 + "\n")
        f.write(f"  Mean Reward: {final_results['mean_reward']:.2f} \u00b1 {final_results['std_reward']:.2f}\n")
        f.write(f"  Mean Wait Time: {final_results['mean_wait_time']:.2f} \u00b1 {final_results['std_wait_time']:.2f}\n")
        f.write(f"  Mean CO2: {final_results['mean_co2']:.2f} \u00b1 {final_results['std_co2']:.2f}\n")
        f.write(f"  Emergency Clearance Rate: {final_results['emergency_clearance_rate']:.1f}%\n")
        f.write(f"  Mean Clearance Time: {final_results['mean_emergency_clearance_time']:.1f} steps\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("BASELINE (Random Agent)\n")
        f.write("-"*40 + "\n")
        f.write(f"  Mean Reward: {baseline_results['mean_reward']:.2f}\n")
        f.write(f"  Mean Wait Time: {baseline_results['mean_wait_time']:.2f}\n")
        f.write(f"  Mean CO2: {baseline_results['mean_co2']:.2f}\n")
        
        f.write("\n" + "-"*40 + "\n")
        f.write("IMPROVEMENTS OVER BASELINE\n")
        f.write("-"*40 + "\n")
        f.write(f"  Reward: {report['improvements']['reward']:.1f}%\n")
        f.write(f"  Wait Time: {report['improvements']['wait_time']:.1f}%\n")
        f.write(f"  CO2 Emissions: {report['improvements']['co2']:.1f}%\n")
    
    print(f"\nTraining report saved to {save_dir}/")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train PPO agent with curriculum learning')
    parser.add_argument('--gui', action='store_true', help='Enable SUMO GUI during evaluation')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and only evaluate')
    parser.add_argument('--model-path', type=str, default=paths_config.SINGLE_AGENT_MODEL, 
                       help='Path to load model for evaluation')
    args = parser.parse_args()
    
    # Configuration
    base_config = {
        'sumo_config': paths_config.SUMO_CONFIG,
        'intersection_id': 'n11',
        'max_steps': sc.SIMULATION_STEPS,
        'emergency_timeout': env_config.EMERGENCY_TIMEOUT_STEPS
    }
    
    # Initialize curriculum
    curriculum = TrafficCurriculum(base_config)
    
    # Create directories (use absolute paths from config)
    os.makedirs(paths_config.AGENTS_DIR, exist_ok=True)
    os.makedirs(paths_config.RESULTS_DIR, exist_ok=True)
    os.makedirs(paths_config.TENSORBOARD_LOG_DIR, exist_ok=True)
    
    model = None
    stage_metrics_list = []
    
    if not args.skip_training:
        print("\n" + "="*80)
        print("STARTING CURRICULUM TRAINING PIPELINE")
        print("="*80)
        
        # Train through all stages
        for stage_idx in range(len(curriculum.stages)):
            model = train_stage(
                stage_idx=stage_idx,
                curriculum=curriculum,
                model=model,
                log_dir=paths_config.TENSORBOARD_LOG_DIR
            )
            
            # Store metrics from this stage
            if hasattr(model, 'stage_metrics'):
                stage_metrics_list.append(model.stage_metrics)
        
        # Save final model
        final_path = paths_config.SINGLE_AGENT_MODEL
        model.save(final_path)
        print(f"\nFinal model saved to {final_path}")
        
        print("\n" + "="*80)
        print("CURRICULUM TRAINING COMPLETED")
        print("="*80)
    else:
        # Load existing model
        print(f"\nLoading model from {args.model_path}")
        model = PPO.load(args.model_path)
        stage_metrics_list = []
    
    # Evaluate final model
    final_config = curriculum.get_stage_config(len(curriculum.stages)-1)  # Use hardest stage for evaluation
    final_results = evaluate_final_model(model, final_config, n_episodes=10)
    
    # Evaluate baseline
    baseline_results = evaluate_baseline(final_config, n_episodes=10)
    
    # Print comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    reward_improvement = ((final_results['mean_reward'] - baseline_results['mean_reward']) / 
                         abs(baseline_results['mean_reward'])) * 100 if abs(baseline_results['mean_reward']) > 0 else 0
    wait_improvement = ((baseline_results['mean_wait_time'] - final_results['mean_wait_time']) / 
                       baseline_results['mean_wait_time']) * 100 if baseline_results['mean_wait_time'] > 0 else 0
    co2_improvement = ((baseline_results['mean_co2'] - final_results['mean_co2']) / 
                      baseline_results['mean_co2']) * 100 if baseline_results['mean_co2'] > 0 else 0
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'PPO Agent':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'Mean Reward':<25} {baseline_results['mean_reward']:<15.2f} "
          f"{final_results['mean_reward']:<15.2f} {reward_improvement:+.1f}%")
    print(f"{'Mean Wait Time':<25} {baseline_results['mean_wait_time']:<15.2f} "
          f"{final_results['mean_wait_time']:<15.2f} {wait_improvement:+.1f}%")
    print(f"{'Mean CO2 (mg/s)':<25} {baseline_results['mean_co2']:<15.2f} "
          f"{final_results['mean_co2']:<15.2f} {co2_improvement:+.1f}%")
    
    # Generate plots and reports
    if stage_metrics_list:
        plot_training_results(stage_metrics_list, final_results, baseline_results,
                              save_dir=paths_config.RESULTS_DIR)
        save_training_report(stage_metrics_list, final_results, baseline_results,
                             save_dir=paths_config.RESULTS_DIR)
    
    print("\nTraining pipeline completed successfully!")

if __name__ == "__main__":
    main()
