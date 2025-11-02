"""Unit tests for RL agents and utilities."""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock, patch
import tempfile
import os

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import load_config, create_env, get_device


class TestDDPGAgent:
    """Test cases for DDPG agent."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = gym.make("Pendulum-v1")
        self.config = {
            "algorithms": {
                "ddpg": {
                    "learning_rate": 1e-3,
                    "buffer_size": 1000,
                    "batch_size": 32,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "exploration_noise": 0.1
                }
            },
            "networks": {
                "actor": {"hidden_sizes": [64, 64]}
            }
        }
        self.agent = DDPGAgent(self.env, self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.state_dim == 3  # Pendulum state dimension
        assert self.agent.action_dim == 1  # Pendulum action dimension
        assert self.agent.max_action == 2.0  # Pendulum max action
        assert len(self.agent.replay_buffer) == 0
    
    def test_action_selection(self):
        """Test action selection."""
        state = np.array([1.0, 0.0, 0.0])
        action = self.agent._select_action(state, add_noise=False)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert -self.agent.max_action <= action[0] <= self.agent.max_action
    
    def test_replay_buffer(self):
        """Test replay buffer functionality."""
        state = np.array([1.0, 0.0, 0.0])
        action = np.array([0.5])
        reward = 1.0
        next_state = np.array([0.9, 0.1, 0.0])
        done = False
        
        self.agent.replay_buffer.add(state, action, reward, next_state, done)
        assert len(self.agent.replay_buffer) == 1
        
        # Test sampling
        batch = self.agent.replay_buffer.sample(1)
        assert len(batch) == 5
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (1, 3)
        assert actions.shape == (1, 1)
        assert rewards.shape == (1, 1)
        assert next_states.shape == (1, 3)
        assert dones.shape == (1, 1)
    
    def test_predict(self):
        """Test prediction method."""
        state = np.array([1.0, 0.0, 0.0])
        action, state_out = self.agent.predict(state, deterministic=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert state_out is None  # DDPG doesn't use hidden states
    
    def test_save_load(self):
        """Test save and load functionality."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            save_path = tmp_file.name
        
        try:
            # Save agent
            self.agent.save(save_path)
            assert os.path.exists(save_path)
            
            # Load agent
            new_agent = DDPGAgent(self.env, self.config)
            new_agent.load(save_path)
            
            # Test that loaded agent works
            state = np.array([1.0, 0.0, 0.0])
            action, _ = new_agent.predict(state)
            assert isinstance(action, np.ndarray)
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestSB3Agents:
    """Test cases for stable-baselines3 agents."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env = gym.make("Pendulum-v1")
        self.config = {
            "algorithms": {
                "ppo": {"learning_rate": 3e-4},
                "sac": {"learning_rate": 3e-4},
                "td3": {"learning_rate": 1e-3},
                "dqn": {"learning_rate": 1e-4}
            }
        }
    
    def test_ppo_agent(self):
        """Test PPO agent."""
        agent = PPOAgent(self.env, self.config)
        assert agent.agent is not None
        
        # Test prediction
        state = np.array([1.0, 0.0, 0.0])
        action, _ = agent.predict(state)
        assert isinstance(action, np.ndarray)
    
    def test_sac_agent(self):
        """Test SAC agent."""
        agent = SACAgent(self.env, self.config)
        assert agent.agent is not None
        
        # Test prediction
        state = np.array([1.0, 0.0, 0.0])
        action, _ = agent.predict(state)
        assert isinstance(action, np.ndarray)
    
    def test_td3_agent(self):
        """Test TD3 agent."""
        agent = TD3Agent(self.env, self.config)
        assert agent.agent is not None
        
        # Test prediction
        state = np.array([1.0, 0.0, 0.0])
        action, _ = agent.predict(state)
        assert isinstance(action, np.ndarray)
    
    def test_dqn_agent(self):
        """Test DQN agent."""
        # DQN requires discrete action space
        env = gym.make("CartPole-v1")
        agent = DQNAgent(env, self.config)
        assert agent.agent is not None
        
        # Test prediction
        state = np.array([0.0, 0.0, 0.0, 0.0])
        action, _ = agent.predict(state)
        assert isinstance(action, np.ndarray)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ["cuda", "cpu"]
    
    def test_create_env(self):
        """Test environment creation."""
        env = create_env("Pendulum-v1")
        assert isinstance(env, gym.Env)
        
        # Test environment step
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        assert isinstance(state, np.ndarray)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create temporary config file
        config_data = {
            "environment": {"name": "Pendulum-v1"},
            "training": {"total_timesteps": 10000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            import yaml
            yaml.dump(config_data, tmp_file)
            config_path = tmp_file.name
        
        try:
            config = load_config(config_path)
            assert config["environment"]["name"] == "Pendulum-v1"
            assert config["training"]["total_timesteps"] == 10000
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestIntegration:
    """Integration tests."""
    
    def test_training_loop(self):
        """Test basic training loop."""
        env = gym.make("Pendulum-v1")
        config = {
            "algorithms": {
                "ddpg": {
                    "learning_rate": 1e-3,
                    "buffer_size": 1000,
                    "batch_size": 32,
                    "gamma": 0.99,
                    "tau": 0.005,
                    "exploration_noise": 0.1
                }
            },
            "networks": {
                "actor": {"hidden_sizes": [64, 64]}
            }
        }
        
        agent = DDPGAgent(env, config)
        
        # Run a few episodes
        for episode in range(3):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(10):  # Short episodes for testing
                action = agent._select_action(state, add_noise=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update networks if buffer has enough samples
            if len(agent.replay_buffer) >= agent.batch_size:
                agent._update_networks()
        
        # Check that agent learned something
        assert len(agent.replay_buffer) > 0
        assert agent.episode_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
