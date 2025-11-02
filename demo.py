#!/usr/bin/env python3
"""Demo script showcasing the RL Robotic Control project."""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import DDPGAgent, PPOAgent
from utils import load_config, create_env, get_device
from utils.visualization import plot_learning_curves, create_training_summary


def demo_ddpg_training():
    """Demonstrate DDPG training on Pendulum environment."""
    print("🤖 DDPG Demo - Training on Pendulum-v1")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Create environment
    env = create_env("Pendulum-v1")
    print(f"Environment: {env.observation_space.shape[0]}D state, {env.action_space.shape[0]}D action")
    
    # Create DDPG agent
    device = get_device()
    print(f"Using device: {device}")
    
    agent = DDPGAgent(env, config, device)
    
    # Train for a short period
    print("Training DDPG agent...")
    agent.learn(total_timesteps=10000, log_interval=5)
    
    # Get training statistics
    stats = agent.get_training_stats()
    print(f"\nTraining completed!")
    print(f"Episodes: {len(stats['episode_rewards'])}")
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Std reward: {stats['std_reward']:.2f}")
    
    # Plot results
    plot_learning_curves(
        rewards=stats['episode_rewards'],
        episode_lengths=stats['episode_lengths'],
        title="DDPG Training Results"
    )
    
    return agent


def demo_ppo_training():
    """Demonstrate PPO training on Pendulum environment."""
    print("\n🎯 PPO Demo - Training on Pendulum-v1")
    print("=" * 50)
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Create environment
    env = create_env("Pendulum-v1")
    
    # Create PPO agent
    device = get_device()
    agent = PPOAgent(env, config, device)
    
    # Train for a short period
    print("Training PPO agent...")
    agent.learn(total_timesteps=10000, log_interval=5)
    
    print("PPO training completed!")
    
    return agent


def demo_algorithm_comparison():
    """Demonstrate algorithm comparison."""
    print("\n⚖️ Algorithm Comparison Demo")
    print("=" * 50)
    
    # Generate sample comparison data
    algorithms = ['DDPG', 'PPO', 'SAC', 'TD3']
    episodes = list(range(1, 101))
    
    comparison_results = {}
    for i, alg in enumerate(algorithms):
        # Generate different learning curves for each algorithm
        base_reward = -200 + i * 50
        rewards = base_reward + np.cumsum(np.random.normal(2, 3, 100)) + np.random.normal(0, 10, 100)
        comparison_results[alg] = {'rewards': rewards}
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange']
    for i, (alg, data) in enumerate(comparison_results.items()):
        rewards = data['rewards']
        plt.plot(episodes, rewards, color=colors[i], linewidth=2, label=alg)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print summary
    print("Algorithm Performance Summary:")
    for alg, data in comparison_results.items():
        rewards = data['rewards']
        print(f"{alg}: Final reward = {rewards[-1]:.2f}, Mean (last 20) = {np.mean(rewards[-20:]):.2f}")


def demo_evaluation():
    """Demonstrate agent evaluation."""
    print("\n📊 Evaluation Demo")
    print("=" * 50)
    
    # Load configuration and create environment
    config = load_config("config/config.yaml")
    env = create_env("Pendulum-v1")
    
    # Create agent
    device = get_device()
    agent = DDPGAgent(env, config, device)
    
    # Evaluate agent
    n_episodes = 10
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating agent for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(200):  # Max episode length
            action = agent._select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Summary:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Best Episode: {max(episode_rewards):.2f}")
    print(f"Worst Episode: {min(episode_rewards):.2f}")
    
    # Plot evaluation results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(episode_rewards, 'o-', color='blue')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(episode_lengths, 'o-', color='orange')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demo function."""
    print("🚀 RL Robotic Control Project Demo")
    print("=" * 60)
    print("This demo showcases the key features of the project:")
    print("1. DDPG training on Pendulum environment")
    print("2. PPO training demonstration")
    print("3. Algorithm comparison")
    print("4. Agent evaluation")
    print("=" * 60)
    
    try:
        # Demo 1: DDPG Training
        ddpg_agent = demo_ddpg_training()
        
        # Demo 2: PPO Training
        ppo_agent = demo_ppo_training()
        
        # Demo 3: Algorithm Comparison
        demo_algorithm_comparison()
        
        # Demo 4: Evaluation
        demo_evaluation()
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("- Use 'python cli.py train --algorithm ddpg --env Pendulum-v1' for CLI training")
        print("- Use 'streamlit run streamlit_app.py' for the web interface")
        print("- Check the README.md for detailed usage instructions")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
