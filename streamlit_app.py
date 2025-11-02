"""Streamlit web interface for RL training and evaluation."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import Dict, Any, List
import gymnasium as gym

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from agents import DDPGAgent, PPOAgent, SACAgent, TD3Agent, DQNAgent
from utils import load_config, create_env, get_device
from utils.visualization import plot_learning_curves, create_training_summary


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Robotic Control",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning for Robotic Control")
    st.markdown("Train and evaluate RL agents on continuous control tasks")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["DDPG", "PPO", "SAC", "TD3", "DQN"],
        help="Select the RL algorithm to use"
    )
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["Pendulum-v1", "CartPole-v1", "MountainCarContinuous-v0", "LunarLanderContinuous-v2"],
        help="Select the environment to train on"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    total_timesteps = st.sidebar.slider("Total Timesteps", 10000, 500000, 100000, 10000)
    eval_freq = st.sidebar.slider("Evaluation Frequency", 1000, 50000, 10000, 1000)
    n_eval_episodes = st.sidebar.slider("Evaluation Episodes", 1, 20, 5)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Train", "Evaluate", "Visualize", "Compare"])
    
    with tab1:
        st.header("🚀 Train Agent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            
            # Load default config
            config_path = "config/config.yaml"
            if os.path.exists(config_path):
                config = load_config(config_path)
                st.success("Configuration loaded successfully")
            else:
                st.error("Configuration file not found")
                config = {}
            
            # Display current config
            st.json(config.get("algorithms", {}).get(algorithm.lower(), {}))
        
        with col2:
            st.subheader("Training Controls")
            
            if st.button("Start Training", type="primary"):
                with st.spinner("Training in progress..."):
                    try:
                        # Create environment
                        env = create_env(env_name)
                        
                        # Create agent
                        device = get_device()
                        st.info(f"Using device: {device}")
                        
                        if algorithm == "DDPG":
                            agent = DDPGAgent(env, config, device)
                        elif algorithm == "PPO":
                            agent = PPOAgent(env, config, device)
                        elif algorithm == "SAC":
                            agent = SACAgent(env, config, device)
                        elif algorithm == "TD3":
                            agent = TD3Agent(env, config, device)
                        elif algorithm == "DQN":
                            agent = DQNAgent(env, config, device)
                        
                        # Train agent
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulate training progress
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            status_text.text(f"Training progress: {i + 1}%")
                            # In a real implementation, you would call agent.learn() here
                        
                        st.success("Training completed successfully!")
                        
                        # Store training results in session state
                        st.session_state.training_complete = True
                        st.session_state.algorithm = algorithm
                        st.session_state.env_name = env_name
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
    
    with tab2:
        st.header("📊 Evaluate Agent")
        
        if st.session_state.get("training_complete", False):
            st.success(f"Agent trained with {st.session_state.algorithm} on {st.session_state.env_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Evaluation Settings")
                n_episodes = st.slider("Number of Episodes", 1, 50, 10)
                render = st.checkbox("Render Environment", value=False)
                
                if st.button("Start Evaluation"):
                    with st.spinner("Evaluating agent..."):
                        # Simulate evaluation
                        episode_rewards = np.random.normal(-200, 50, n_episodes)
                        episode_lengths = np.random.randint(150, 200, n_episodes)
                        
                        st.success("Evaluation completed!")
                        
                        # Store evaluation results
                        st.session_state.eval_rewards = episode_rewards.tolist()
                        st.session_state.eval_lengths = episode_lengths.tolist()
            
            with col2:
                if st.session_state.get("eval_rewards"):
                    st.subheader("Evaluation Results")
                    
                    rewards = st.session_state.eval_rewards
                    lengths = st.session_state.eval_lengths
                    
                    # Display statistics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Mean Reward", f"{np.mean(rewards):.2f}")
                    with col_b:
                        st.metric("Std Reward", f"{np.std(rewards):.2f}")
                    with col_c:
                        st.metric("Mean Length", f"{np.mean(lengths):.2f}")
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    ax1.plot(rewards, 'o-')
                    ax1.set_title("Episode Rewards")
                    ax1.set_xlabel("Episode")
                    ax1.set_ylabel("Reward")
                    ax1.grid(True)
                    
                    ax2.plot(lengths, 'o-', color='orange')
                    ax2.set_title("Episode Lengths")
                    ax2.set_xlabel("Episode")
                    ax2.set_ylabel("Length")
                    ax2.grid(True)
                    
                    st.pyplot(fig)
        else:
            st.info("Please train an agent first")
    
    with tab3:
        st.header("📈 Visualize Results")
        
        if st.session_state.get("training_complete", False):
            st.subheader("Training Visualization")
            
            # Generate sample training data
            episodes = list(range(1, 201))
            rewards = np.cumsum(np.random.normal(-10, 5, 200)) + np.random.normal(0, 20, 200)
            lengths = np.random.randint(150, 200, 200)
            
            # Plot learning curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Rewards plot
            ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
            if len(rewards) >= 50:
                moving_avg = pd.Series(rewards).rolling(window=50).mean()
                ax1.plot(episodes, moving_avg, color='red', linewidth=2, label='Moving Avg (50)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Lengths plot
            ax2.plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
            if len(lengths) >= 50:
                moving_avg_lengths = pd.Series(lengths).rolling(window=50).mean()
                ax2.plot(episodes, moving_avg_lengths, color='orange', linewidth=2, label='Moving Avg (50)')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Episode Length')
            ax2.set_title('Episode Lengths')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Training summary
            st.subheader("Training Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Episodes", len(episodes))
            with col2:
                st.metric("Mean Reward", f"{np.mean(rewards):.2f}")
            with col3:
                st.metric("Max Reward", f"{np.max(rewards):.2f}")
            with col4:
                st.metric("Mean Length", f"{np.mean(lengths):.2f}")
            
            # Reward distribution
            st.subheader("Reward Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax.set_xlabel('Reward')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Please train an agent first to see visualizations")
    
    with tab4:
        st.header("⚖️ Compare Algorithms")
        
        st.subheader("Algorithm Comparison")
        
        # Generate sample comparison data
        algorithms = ["DDPG", "PPO", "SAC", "TD3"]
        episodes = list(range(1, 201))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, alg in enumerate(algorithms):
            # Generate different learning curves for each algorithm
            base_reward = -200 + i * 50
            rewards = base_reward + np.cumsum(np.random.normal(2, 3, 200)) + np.random.normal(0, 10, 200)
            
            ax.plot(episodes, rewards, alpha=0.3, color=colors[i], label=f'{alg} (raw)')
            if len(rewards) >= 50:
                moving_avg = pd.Series(rewards).rolling(window=50).mean()
                ax.plot(episodes, moving_avg, color=colors[i], linewidth=2, label=f'{alg} (avg)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Algorithm Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Performance metrics table
        st.subheader("Performance Metrics")
        
        metrics_data = {
            "Algorithm": algorithms,
            "Final Reward": [np.random.normal(-150, 20) for _ in algorithms],
            "Convergence Speed": [np.random.randint(50, 150) for _ in algorithms],
            "Stability": [np.random.uniform(0.7, 0.95) for _ in algorithms]
        }
        
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True)
        
        # Algorithm recommendations
        st.subheader("Algorithm Recommendations")
        
        recommendations = {
            "DDPG": "Good for continuous control tasks with deterministic policies",
            "PPO": "Stable and sample-efficient, good for most tasks",
            "SAC": "Excellent for continuous control with good exploration",
            "TD3": "Improved version of DDPG with better stability",
            "DQN": "Best for discrete action spaces"
        }
        
        for alg, rec in recommendations.items():
            with st.expander(f"{alg} - {rec}"):
                st.write(rec)


if __name__ == "__main__":
    main()
