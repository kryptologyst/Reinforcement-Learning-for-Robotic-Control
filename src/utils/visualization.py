"""Visualization utilities for RL training."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd


def plot_learning_curves(
    rewards: List[float],
    episode_lengths: Optional[List[int]] = None,
    window_size: int = 100,
    title: str = "Learning Curves",
    save_path: Optional[str] = None
) -> None:
    """Plot learning curves for rewards and episode lengths.
    
    Args:
        rewards: List of episode rewards
        episode_lengths: List of episode lengths (optional)
        window_size: Window size for moving average
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2 if episode_lengths else 1, figsize=(12, 5))
    if episode_lengths is None:
        axes = [axes]
    
    # Plot rewards
    episodes = range(1, len(rewards) + 1)
    axes[0].plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    
    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        axes[0].plot(episodes, moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot episode lengths if provided
    if episode_lengths:
        axes[1].plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
        
        if len(episode_lengths) >= window_size:
            moving_avg_lengths = pd.Series(episode_lengths).rolling(window=window_size).mean()
            axes[1].plot(episodes, moving_avg_lengths, color='orange', linewidth=2, label=f'Moving Avg ({window_size})')
        
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Lengths')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_algorithm_comparison(
    results: Dict[str, Dict[str, List[float]]],
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None
) -> None:
    """Plot comparison of different algorithms.
    
    Args:
        results: Dictionary with algorithm names as keys and reward lists as values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for alg_name, data in results.items():
        rewards = data.get('rewards', [])
        episodes = range(1, len(rewards) + 1)
        
        # Plot raw rewards with transparency
        plt.plot(episodes, rewards, alpha=0.2, label=f'{alg_name} (raw)')
        
        # Plot moving average
        if len(rewards) >= 50:
            moving_avg = pd.Series(rewards).rolling(window=50).mean()
            plt.plot(episodes, moving_avg, linewidth=2, label=f'{alg_name} (avg)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_sensitivity(
    results: Dict[str, List[float]],
    param_name: str,
    param_values: List[Any],
    title: str = "Hyperparameter Sensitivity",
    save_path: Optional[str] = None
) -> None:
    """Plot hyperparameter sensitivity analysis.
    
    Args:
        results: Dictionary with parameter values as keys and reward lists as values
        param_name: Name of the parameter
        param_values: List of parameter values
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, (param_val, rewards) in enumerate(results.items()):
        episodes = range(1, len(rewards) + 1)
        
        # Plot moving average
        if len(rewards) >= 20:
            moving_avg = pd.Series(rewards).rolling(window=20).mean()
            plt.plot(episodes, moving_avg, linewidth=2, label=f'{param_name}={param_val}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reward_distribution(
    rewards: List[float],
    bins: int = 50,
    title: str = "Reward Distribution",
    save_path: Optional[str] = None
) -> None:
    """Plot reward distribution histogram.
    
    Args:
        rewards: List of episode rewards
        bins: Number of bins for histogram
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(rewards, bins=bins, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_summary(
    stats: Dict[str, Any],
    save_path: Optional[str] = None
) -> None:
    """Create a comprehensive training summary plot.
    
    Args:
        stats: Dictionary containing training statistics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    rewards = stats.get('rewards', [])
    if rewards:
        episodes = range(1, len(rewards) + 1)
        axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue')
        
        if len(rewards) >= 50:
            moving_avg = pd.Series(rewards).rolling(window=50).mean()
            axes[0, 0].plot(episodes, moving_avg, color='red', linewidth=2)
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    lengths = stats.get('episode_lengths', [])
    if lengths:
        episodes = range(1, len(lengths) + 1)
        axes[0, 1].plot(episodes, lengths, alpha=0.3, color='green')
        
        if len(lengths) >= 50:
            moving_avg = pd.Series(lengths).rolling(window=50).mean()
            axes[0, 1].plot(episodes, moving_avg, color='orange', linewidth=2)
        
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot reward distribution
    if rewards:
        axes[1, 0].hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot training statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Training Statistics:
    
    Total Episodes: {len(rewards)}
    Mean Reward: {np.mean(rewards):.2f}
    Std Reward: {np.std(rewards):.2f}
    Max Reward: {np.max(rewards):.2f}
    Min Reward: {np.min(rewards):.2f}
    
    Mean Episode Length: {np.mean(lengths):.2f}
    Std Episode Length: {np.std(lengths):.2f}
    """
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                     fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Training Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
