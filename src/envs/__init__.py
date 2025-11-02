"""Custom environment wrappers and utilities."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Any


class RewardShapingWrapper(gym.Wrapper):
    """Wrapper that adds reward shaping to an environment."""
    
    def __init__(self, env: gym.Env, reward_scale: float = 1.0):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            reward_scale: Scaling factor for rewards
        """
        super().__init__(env)
        self.reward_scale = reward_scale
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Step the environment with reward shaping."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward * self.reward_scale
        
        return observation, shaped_reward, terminated, truncated, info


class ActionNoiseWrapper(gym.Wrapper):
    """Wrapper that adds noise to actions."""
    
    def __init__(self, env: gym.Env, noise_std: float = 0.1):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            noise_std: Standard deviation of action noise
        """
        super().__init__(env)
        self.noise_std = noise_std
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Step the environment with action noise."""
        # Add noise to action
        if isinstance(action, np.ndarray):
            noise = np.random.normal(0, self.noise_std, action.shape)
            noisy_action = np.clip(action + noise, self.action_space.low, self.action_space.high)
        else:
            noisy_action = action
        
        return self.env.step(noisy_action)


class StateNormalizationWrapper(gym.Wrapper):
    """Wrapper that normalizes state observations."""
    
    def __init__(self, env: gym.Env):
        """Initialize the wrapper."""
        super().__init__(env)
        self.state_mean = None
        self.state_std = None
        self.state_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        """Reset the environment and update normalization statistics."""
        observation, info = self.env.reset(seed=seed, options=options)
        
        # Update normalization statistics
        self._update_normalization_stats(observation)
        
        # Normalize observation
        normalized_observation = self._normalize_observation(observation)
        
        return normalized_observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Step the environment with normalized observations."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Update normalization statistics
        self._update_normalization_stats(observation)
        
        # Normalize observation
        normalized_observation = self._normalize_observation(observation)
        
        return normalized_observation, reward, terminated, truncated, info
    
    def _update_normalization_stats(self, observation: np.ndarray) -> None:
        """Update running statistics for normalization."""
        if self.state_mean is None:
            self.state_mean = np.zeros_like(observation)
            self.state_std = np.ones_like(observation)
        
        self.state_count += 1
        
        # Update running mean and std
        delta = observation - self.state_mean
        self.state_mean += delta / self.state_count
        
        if self.state_count > 1:
            delta2 = observation - self.state_mean
            self.state_std = np.sqrt(
                (self.state_std ** 2 * (self.state_count - 1) + delta2 ** 2) / self.state_count
            )
    
    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        if self.state_mean is None:
            return observation
        
        # Avoid division by zero
        std = np.where(self.state_std < 1e-8, 1.0, self.state_std)
        
        return (observation - self.state_mean) / std


class EpisodeLoggerWrapper(gym.Wrapper):
    """Wrapper that logs episode statistics."""
    
    def __init__(self, env: gym.Env):
        """Initialize the wrapper."""
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        """Reset the environment and log episode statistics."""
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        observation, info = self.env.reset(seed=seed, options=options)
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Step the environment and update episode statistics."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        return observation, reward, terminated, truncated, info
    
    def get_episode_stats(self) -> dict:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {"mean_reward": 0, "std_reward": 0, "mean_length": 0, "std_length": 0}
        
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "num_episodes": len(self.episode_rewards)
        }


def create_wrapped_env(env_name: str, wrappers: list = None) -> gym.Env:
    """Create an environment with specified wrappers.
    
    Args:
        env_name: Name of the environment
        wrappers: List of wrapper classes to apply
        
    Returns:
        Wrapped environment
    """
    env = gym.make(env_name)
    
    if wrappers:
        for wrapper in wrappers:
            env = wrapper(env)
    
    return env


# Example usage
if __name__ == "__main__":
    # Create environment with multiple wrappers
    env = create_wrapped_env(
        "Pendulum-v1",
        wrappers=[
            RewardShapingWrapper,
            ActionNoiseWrapper,
            StateNormalizationWrapper,
            EpisodeLoggerWrapper
        ]
    )
    
    # Test the wrapped environment
    observation, _ = env.reset()
    print(f"Initial observation: {observation}")
    
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            break
    
    # Get episode statistics
    stats = env.get_episode_stats()
    print(f"Episode statistics: {stats}")
