"""Base agent class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(
        self,
        env: Union[gym.Env, VecEnv],
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the agent.
        
        Args:
            env: The environment to train on
            config: Configuration dictionary
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.env = env
        self.config = config
        self.device = self._get_device(device)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for computation."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
        eval_env: Optional[Union[gym.Env, VecEnv]] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "BaseAgent":
        """Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback function called during training
            log_interval: Log every log_interval timesteps
            eval_env: Environment for evaluation
            eval_freq: Evaluate every eval_freq timesteps
            n_eval_episodes: Number of episodes to evaluate
            eval_log_path: Path to save evaluation logs
            reset_num_timesteps: Whether to reset timestep counter
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Predict action given observation.
        
        Args:
            observation: Current observation
            state: Hidden state (for recurrent policies)
            episode_start: Whether this is the start of an episode
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and next state
        """
        pass
    
    def save(self, path: str) -> None:
        """Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        raise NotImplementedError("Subclasses must implement save method")
    
    def load(self, path: str) -> None:
        """Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        raise NotImplementedError("Subclasses must implement load method")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0.0,
        }
