"""Modern RL agents using stable-baselines3."""

from typing import Any, Dict, Optional, Union
import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent using stable-baselines3."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the PPO agent.
        
        Args:
            env: The environment to train on
            config: Configuration dictionary
            device: Device to run on
        """
        super().__init__(env, config, device)
        
        ppo_config = config.get("algorithms", {}).get("ppo", {})
        
        self.agent = PPO(
            "MlpPolicy",
            env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            device=device,
            verbose=1
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 4,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPOAgent":
        """Train the PPO agent."""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps
        )
        return self
    
    def predict(
        self,
        observation: Union[Any, Dict[str, Any]],
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> tuple:
        """Predict action given observation."""
        return self.agent.predict(observation, state, episode_start, deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent from a file."""
        self.agent = PPO.load(path)


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent using stable-baselines3."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the SAC agent."""
        super().__init__(env, config, device)
        
        sac_config = config.get("algorithms", {}).get("sac", {})
        
        self.agent = SAC(
            "MlpPolicy",
            env,
            learning_rate=sac_config.get("learning_rate", 3e-4),
            buffer_size=sac_config.get("buffer_size", 100000),
            batch_size=sac_config.get("batch_size", 256),
            gamma=sac_config.get("gamma", 0.99),
            tau=sac_config.get("tau", 0.005),
            target_update_interval=sac_config.get("target_update_interval", 1),
            device=device,
            verbose=1
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 4,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "SACAgent":
        """Train the SAC agent."""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps
        )
        return self
    
    def predict(
        self,
        observation: Union[Any, Dict[str, Any]],
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> tuple:
        """Predict action given observation."""
        return self.agent.predict(observation, state, episode_start, deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent from a file."""
        self.agent = SAC.load(path)


class TD3Agent(BaseAgent):
    """Twin Delayed Deep Deterministic Policy Gradient agent using stable-baselines3."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the TD3 agent."""
        super().__init__(env, config, device)
        
        td3_config = config.get("algorithms", {}).get("td3", {})
        
        self.agent = TD3(
            "MlpPolicy",
            env,
            learning_rate=td3_config.get("learning_rate", 1e-3),
            buffer_size=td3_config.get("buffer_size", 100000),
            batch_size=td3_config.get("batch_size", 256),
            gamma=td3_config.get("gamma", 0.99),
            tau=td3_config.get("tau", 0.005),
            policy_delay=td3_config.get("policy_delay", 2),
            target_noise_clip=td3_config.get("target_noise_clip", 0.5),
            target_noise=td3_config.get("target_noise", 0.2),
            device=device,
            verbose=1
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 4,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "TD3Agent":
        """Train the TD3 agent."""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps
        )
        return self
    
    def predict(
        self,
        observation: Union[Any, Dict[str, Any]],
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> tuple:
        """Predict action given observation."""
        return self.agent.predict(observation, state, episode_start, deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent from a file."""
        self.agent = TD3.load(path)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent using stable-baselines3."""
    
    def __init__(
        self,
        env: Union[gym.Env, str],
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the DQN agent."""
        super().__init__(env, config, device)
        
        dqn_config = config.get("algorithms", {}).get("dqn", {})
        
        self.agent = DQN(
            "MlpPolicy",
            env,
            learning_rate=dqn_config.get("learning_rate", 1e-4),
            buffer_size=dqn_config.get("buffer_size", 50000),
            batch_size=dqn_config.get("batch_size", 32),
            gamma=dqn_config.get("gamma", 0.99),
            tau=dqn_config.get("tau", 1.0),
            target_update_interval=dqn_config.get("target_update_interval", 10000),
            exploration_fraction=dqn_config.get("exploration_fraction", 0.1),
            exploration_initial_eps=dqn_config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=dqn_config.get("exploration_final_eps", 0.05),
            device=device,
            verbose=1
        )
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 4,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DQNAgent":
        """Train the DQN agent."""
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps
        )
        return self
    
    def predict(
        self,
        observation: Union[Any, Dict[str, Any]],
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = False,
    ) -> tuple:
        """Predict action given observation."""
        return self.agent.predict(observation, state, episode_start, deterministic)
    
    def save(self, path: str) -> None:
        """Save the agent to a file."""
        self.agent.save(path)
    
    def load(self, path: str) -> None:
        """Load the agent from a file."""
        self.agent = DQN.load(path)
