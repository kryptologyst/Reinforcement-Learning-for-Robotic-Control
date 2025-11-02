"""RL Robotic Control - A modern reinforcement learning project."""

__version__ = "1.0.0"
__author__ = "RL Robotic Control Team"
__description__ = "A comprehensive RL project for continuous control tasks"

from .agents import DDPGAgent, PPOAgent, SACAgent, TD3Agent, DQNAgent
from .utils import load_config, create_env, get_device

__all__ = [
    "DDPGAgent",
    "PPOAgent", 
    "SACAgent",
    "TD3Agent",
    "DQNAgent",
    "load_config",
    "create_env",
    "get_device"
]
