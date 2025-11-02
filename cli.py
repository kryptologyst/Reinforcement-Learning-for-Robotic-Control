#!/usr/bin/env python3
"""Command-line interface for RL training and evaluation."""

import argparse
import os
import sys
from typing import Dict, Any
import gymnasium as gym

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import load_config, create_env, create_vec_env, create_eval_callback, ensure_dir, get_device
from utils.visualization import plot_learning_curves, create_training_summary


def train_agent(
    algorithm: str,
    env_name: str,
    config_path: str,
    total_timesteps: int,
    save_path: str,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    render: bool = False
) -> None:
    """Train an RL agent.
    
    Args:
        algorithm: Algorithm to use (ddpg, ppo, sac, td3, dqn)
        env_name: Name of the environment
        config_path: Path to configuration file
        total_timesteps: Total timesteps to train
        save_path: Path to save the trained model
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        render: Whether to render during training
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = create_env(env_name, render_mode)
    
    # Create evaluation environment
    eval_env = create_env(env_name)
    
    # Create agent
    device = get_device()
    print(f"Using device: {device}")
    
    if algorithm.lower() == "ddpg":
        agent = DDPGAgent(env, config, device)
    elif algorithm.lower() == "ppo":
        agent = PPOAgent(env, config, device)
    elif algorithm.lower() == "sac":
        agent = SACAgent(env, config, device)
    elif algorithm.lower() == "td3":
        agent = TD3Agent(env, config, device)
    elif algorithm.lower() == "dqn":
        agent = DQNAgent(env, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create evaluation callback
    eval_callback = create_eval_callback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        log_path=f"logs/{algorithm}_eval",
        best_model_save_path=f"checkpoints/{algorithm}_best"
    )
    
    # Ensure directories exist
    ensure_dir("logs")
    ensure_dir("checkpoints")
    
    print(f"Training {algorithm.upper()} agent on {env_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Evaluation frequency: {eval_freq}")
    
    # Train agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes
    )
    
    # Save final model
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    # Plot training results
    stats = agent.get_training_stats()
    plot_learning_curves(
        rewards=stats['episode_rewards'],
        episode_lengths=stats['episode_lengths'],
        title=f"{algorithm.upper()} Training Results"
    )
    
    create_training_summary(stats)


def evaluate_agent(
    algorithm: str,
    env_name: str,
    model_path: str,
    n_episodes: int = 10,
    render: bool = True
) -> None:
    """Evaluate a trained agent.
    
    Args:
        algorithm: Algorithm used (ddpg, ppo, sac, td3, dqn)
        env_name: Name of the environment
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
    """
    # Create environment
    render_mode = "human" if render else None
    env = create_env(env_name, render_mode)
    
    # Load agent
    device = get_device()
    config = {"algorithms": {}, "networks": {}}
    
    if algorithm.lower() == "ddpg":
        agent = DDPGAgent(env, config, device)
    elif algorithm.lower() == "ppo":
        agent = PPOAgent(env, config, device)
    elif algorithm.lower() == "sac":
        agent = SACAgent(env, config, device)
    elif algorithm.lower() == "td3":
        agent = TD3Agent(env, config, device)
    elif algorithm.lower() == "dqn":
        agent = DQNAgent(env, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    agent.load(model_path)
    
    # Evaluate agent
    print(f"Evaluating {algorithm.upper()} agent on {env_name}")
    print(f"Number of episodes: {n_episodes}")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print summary statistics
    print(f"\nEvaluation Summary:")
    print(f"Mean Reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
    print(f"Std Reward: {(sum([(r - sum(episode_rewards) / len(episode_rewards))**2 for r in episode_rewards]) / len(episode_rewards))**0.5:.2f}")
    print(f"Mean Episode Length: {sum(episode_lengths) / len(episode_lengths):.2f}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RL Training and Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train an RL agent")
    train_parser.add_argument("--algorithm", required=True, choices=["ddpg", "ppo", "sac", "td3", "dqn"],
                             help="RL algorithm to use")
    train_parser.add_argument("--env", required=True, help="Environment name")
    train_parser.add_argument("--config", required=True, help="Path to configuration file")
    train_parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train")
    train_parser.add_argument("--save-path", required=True, help="Path to save the trained model")
    train_parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    train_parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    train_parser.add_argument("--render", action="store_true", help="Render during training")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("--algorithm", required=True, choices=["ddpg", "ppo", "sac", "td3", "dqn"],
                           help="RL algorithm used")
    eval_parser.add_argument("--env", required=True, help="Environment name")
    eval_parser.add_argument("--model-path", required=True, help="Path to the trained model")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    eval_parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_agent(
            algorithm=args.algorithm,
            env_name=args.env,
            config_path=args.config,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            render=args.render
        )
    elif args.command == "evaluate":
        evaluate_agent(
            algorithm=args.algorithm,
            env_name=args.env,
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=args.render
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
