"""Utility functions for the RL project."""

import os
import yaml
from typing import Any, Dict, Optional
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_env(env_name: str, render_mode: Optional[str] = None) -> gym.Env:
    """Create and wrap environment.
    
    Args:
        env_name: Name of the environment
        render_mode: Render mode for visualization
        
    Returns:
        Wrapped environment
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)
    return env


def create_vec_env(
    env_name: str,
    n_envs: int = 1,
    render_mode: Optional[str] = None,
    normalize: bool = True
) -> VecNormalize:
    """Create vectorized environment.
    
    Args:
        env_name: Name of the environment
        n_envs: Number of parallel environments
        render_mode: Render mode for visualization
        normalize: Whether to normalize observations and rewards
        
    Returns:
        Vectorized environment
    """
    def make_env():
        return gym.make(env_name, render_mode=render_mode)
    
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    return env


def create_eval_callback(
    eval_env: gym.Env,
    eval_freq: int,
    n_eval_episodes: int = 5,
    log_path: Optional[str] = None,
    best_model_save_path: Optional[str] = None,
    deterministic: bool = True,
    render: bool = False
) -> EvalCallback:
    """Create evaluation callback.
    
    Args:
        eval_env: Environment for evaluation
        eval_freq: Evaluate every eval_freq timesteps
        n_eval_episodes: Number of episodes to evaluate
        log_path: Path to save evaluation logs
        best_model_save_path: Path to save best model
        deterministic: Whether to use deterministic policy
        render: Whether to render during evaluation
        
    Returns:
        Evaluation callback
    """
    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        verbose=1
    )


def create_stop_callback(reward_threshold: float) -> StopTrainingOnRewardThreshold:
    """Create stop training callback.
    
    Args:
        reward_threshold: Reward threshold to stop training
        
    Returns:
        Stop training callback
    """
    return StopTrainingOnRewardThreshold(reward_threshold=reward_threshold)


def ensure_dir(path: str) -> None:
    """Ensure directory exists.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_device() -> str:
    """Get the best available device.
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
