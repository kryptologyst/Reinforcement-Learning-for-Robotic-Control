# Reinforcement Learning for Robotic Control

A comprehensive reinforcement learning project focused on continuous control tasks for robotics. This project implements state-of-the-art RL algorithms with a clean, modular architecture and provides both CLI and web interfaces for training and evaluation.

## Features

- **Multiple RL Algorithms**: DDPG, PPO, SAC, TD3, and DQN implementations
- **Modern Libraries**: Uses Gymnasium, Stable-Baselines3, PyTorch, and NumPy
- **Clean Architecture**: Modular design with separate agents, environments, and utilities
- **Type Hints**: Full type annotations for better code quality
- **Configuration Management**: YAML-based configuration system
- **Visualization**: Comprehensive plotting and analysis tools
- **Web Interface**: Streamlit-based interactive dashboard
- **CLI Interface**: Command-line tools for training and evaluation
- **Testing**: Comprehensive unit tests for all components
- **Logging**: TensorBoard and Weights & Biases integration

## 📁 Project Structure

```
0265_RL_for_robotic_control/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── ddpg_agent.py
│   │   └── sb3_agents.py
│   ├── envs/
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── config/
│   └── config.yaml
├── tests/
│   └── test_agents.py
├── notebooks/
├── logs/
├── checkpoints/
├── cli.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Reinforcement-Learning-for-Robotic-Control.git
   cd Reinforcement-Learning-for-Robotic-Control
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Using the CLI

1. **Train an agent**:
   ```bash
   python cli.py train --algorithm ddpg --env Pendulum-v1 --config config/config.yaml --timesteps 100000 --save-path checkpoints/ddpg_model
   ```

2. **Evaluate a trained agent**:
   ```bash
   python cli.py evaluate --algorithm ddpg --env Pendulum-v1 --model-path checkpoints/ddpg_model --episodes 10 --render
   ```

### Using the Web Interface

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

### Using Python API

```python
from src.agents import DDPGAgent
from src.utils import load_config, create_env

# Load configuration
config = load_config("config/config.yaml")

# Create environment
env = create_env("Pendulum-v1")

# Create and train agent
agent = DDPGAgent(env, config)
agent.learn(total_timesteps=100000)

# Evaluate agent
state, _ = env.reset()
action, _ = agent.predict(state, deterministic=True)
```

## Supported Algorithms

### Deep Deterministic Policy Gradient (DDPG)
- **Best for**: Continuous control tasks
- **Key features**: Actor-critic architecture, experience replay, target networks
- **Use case**: Robotic arm control, autonomous vehicles

### Proximal Policy Optimization (PPO)
- **Best for**: General-purpose RL tasks
- **Key features**: Policy gradient method, clipping for stability
- **Use case**: General robotics, game playing

### Soft Actor-Critic (SAC)
- **Best for**: Continuous control with exploration
- **Key features**: Maximum entropy RL, automatic temperature tuning
- **Use case**: Dexterous manipulation, locomotion

### Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **Best for**: Continuous control with improved stability
- **Key features**: Twin critics, delayed policy updates, target policy smoothing
- **Use case**: High-dimensional continuous control

### Deep Q-Network (DQN)
- **Best for**: Discrete action spaces
- **Key features**: Q-learning with neural networks, experience replay
- **Use case**: Discrete decision making, game playing

## Supported Environments

- **Pendulum-v1**: Classic continuous control task
- **CartPole-v1**: Discrete control benchmark
- **MountainCarContinuous-v0**: Continuous mountain car
- **LunarLanderContinuous-v2**: Continuous lunar lander
- **Custom environments**: Easy to add your own

## Configuration

The project uses YAML configuration files. Key settings include:

```yaml
# Environment settings
environment:
  name: "Pendulum-v1"
  max_episode_steps: 200

# Training settings
training:
  total_timesteps: 100000
  eval_freq: 10000
  save_freq: 20000

# Algorithm-specific hyperparameters
algorithms:
  ddpg:
    learning_rate: 1e-3
    buffer_size: 100000
    batch_size: 128
    gamma: 0.99
    tau: 0.005
    exploration_noise: 0.1
```

## Visualization and Analysis

The project includes comprehensive visualization tools:

- **Learning curves**: Episode rewards and lengths over time
- **Algorithm comparison**: Side-by-side performance comparison
- **Hyperparameter sensitivity**: Analysis of parameter effects
- **Reward distributions**: Statistical analysis of performance
- **Training summaries**: Comprehensive training statistics

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_agents.py
```

## Monitoring and Logging

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases
Enable in configuration:
```yaml
logging:
  wandb: true
  wandb_project: "rl-robotic-control"
```

## 🔧 Advanced Usage

### Custom Environments

Create your own environment by extending Gymnasium:

```python
import gymnasium as gym
from gymnasium import spaces

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,))
    
    def step(self, action):
        # Your environment logic
        pass
    
    def reset(self, seed=None):
        # Your reset logic
        pass
```

### Custom Agents

Extend the base agent class:

```python
from src.agents import BaseAgent

class CustomAgent(BaseAgent):
    def learn(self, total_timesteps, **kwargs):
        # Your training logic
        pass
    
    def predict(self, observation, **kwargs):
        # Your prediction logic
        pass
```

## Examples

### Basic Training Example

```python
from src.agents import PPOAgent
from src.utils import load_config, create_env

# Load configuration
config = load_config("config/config.yaml")

# Create environment
env = create_env("Pendulum-v1")

# Create agent
agent = PPOAgent(env, config)

# Train
agent.learn(total_timesteps=100000)

# Save model
agent.save("checkpoints/ppo_model")
```

### Hyperparameter Tuning

```python
import itertools

# Define parameter grid
learning_rates = [1e-4, 3e-4, 1e-3]
batch_sizes = [32, 64, 128]

best_reward = float('-inf')
best_params = None

for lr, batch_size in itertools.product(learning_rates, batch_sizes):
    config["algorithms"]["ppo"]["learning_rate"] = lr
    config["algorithms"]["ppo"]["batch_size"] = batch_size
    
    agent = PPOAgent(env, config)
    agent.learn(total_timesteps=50000)
    
    # Evaluate
    rewards = []
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        for _ in range(200):
            action, _ = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    
    mean_reward = sum(rewards) / len(rewards)
    if mean_reward > best_reward:
        best_reward = mean_reward
        best_params = (lr, batch_size)

print(f"Best parameters: {best_params}, Best reward: {best_reward}")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Gym](https://gym.openai.com/) for the environment framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for RL algorithms
- [PyTorch](https://pytorch.org/) for deep learning
- [Streamlit](https://streamlit.io/) for the web interface

## Future Enhancements

- [ ] Multi-agent RL support
- [ ] Distributed training with Ray
- [ ] More sophisticated visualization tools
- [ ] Integration with ROS (Robot Operating System)
- [ ] Real robot deployment examples
- [ ] Advanced curriculum learning
- [ ] Meta-learning capabilities

 
# Reinforcement-Learning-for-Robotic-Control
