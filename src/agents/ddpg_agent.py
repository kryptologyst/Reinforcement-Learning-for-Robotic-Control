"""Deep Deterministic Policy Gradient (DDPG) agent implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from .base_agent import BaseAgent


class Actor(nn.Module):
    """Actor network for DDPG."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_sizes: List[int] = [256, 256]
    ) -> None:
        """Initialize the actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        self.max_action = max_action
        
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network.
        
        Args:
            state: Current state
            
        Returns:
            Action values scaled by max_action
        """
        return self.max_action * self.network(state)


class Critic(nn.Module):
    """Critic network for DDPG."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256]
    ) -> None:
        """Initialize the critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
        """
        super().__init__()
        
        layers = []
        prev_size = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network.
        
        Args:
            state: Current state
            action: Current action
            
        Returns:
            Q-value estimate
        """
        return self.network(torch.cat([state, action], dim=1))


class ReplayBuffer:
    """Experience replay buffer for DDPG."""
    
    def __init__(self, capacity: int = 100000) -> None:
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DDPGAgent(BaseAgent):
    """DDPG agent implementation."""
    
    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        device: str = "auto"
    ) -> None:
        """Initialize the DDPG agent.
        
        Args:
            env: The environment to train on
            config: Configuration dictionary
            device: Device to run on
        """
        super().__init__(env, config, device)
        
        # Extract configuration
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        
        ddpg_config = config.get("algorithms", {}).get("ddpg", {})
        self.learning_rate = ddpg_config.get("learning_rate", 1e-3)
        self.buffer_size = ddpg_config.get("buffer_size", 100000)
        self.batch_size = ddpg_config.get("batch_size", 128)
        self.gamma = ddpg_config.get("gamma", 0.99)
        self.tau = ddpg_config.get("tau", 0.005)
        self.exploration_noise = ddpg_config.get("exploration_noise", 0.1)
        
        network_config = config.get("networks", {})
        hidden_sizes = network_config.get("actor", {}).get("hidden_sizes", [256, 256])
        
        # Initialize networks
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action, hidden_sizes)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action, hidden_sizes)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.state_dim, self.action_dim, hidden_sizes)
        self.critic_target = Critic(self.state_dim, self.action_dim, hidden_sizes)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Move to device
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Training statistics
        self.num_timesteps = 0
        self.episode_count = 0
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DDPGAgent":
        """Train the DDPG agent.
        
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
        if reset_num_timesteps:
            self.num_timesteps = 0
            self.episode_count = 0
        
        while self.num_timesteps < total_timesteps:
            episode_reward, episode_length = self._train_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1
            
            if callback is not None:
                callback.locals["episode_reward"] = episode_reward
                callback.locals["episode_length"] = episode_length
                callback.on_step()
            
            if log_interval > 0 and self.episode_count % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-log_interval:])
                print(f"Episode {self.episode_count}, Mean Reward: {mean_reward:.2f}")
        
        return self
    
    def _train_episode(self) -> Tuple[float, int]:
        """Train for one episode.
        
        Returns:
            Episode reward and length
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(200):  # Max episode length
            action = self._select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.num_timesteps += 1
            
            if len(self.replay_buffer) >= self.batch_size:
                self._update_networks()
            
            if done:
                break
        
        return episode_reward, episode_length
    
    def _select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select action given state.
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.exploration_noise * np.random.randn(self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def _update_networks(self) -> None:
        """Update actor and critic networks."""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        """Soft update target network.
        
        Args:
            target: Target network
            source: Source network
            tau: Soft update coefficient
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
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
            state: Hidden state (not used in DDPG)
            episode_start: Whether this is the start of an episode
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action and next state
        """
        if isinstance(observation, dict):
            observation = observation["observation"]
        
        action = self._select_action(observation, add_noise=not deterministic)
        return action, None
    
    def save(self, path: str) -> None:
        """Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
            "num_timesteps": self.num_timesteps,
            "episode_count": self.episode_count,
        }, path)
    
    def load(self, path: str) -> None:
        """Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.num_timesteps = checkpoint.get("num_timesteps", 0)
        self.episode_count = checkpoint.get("episode_count", 0)
