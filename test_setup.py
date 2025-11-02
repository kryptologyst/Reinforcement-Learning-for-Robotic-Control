#!/usr/bin/env python3
"""Test script to verify the RL project setup."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from agents import DDPGAgent, PPOAgent, SACAgent, TD3Agent, DQNAgent
        print("✅ Agent imports successful")
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    try:
        from utils import load_config, create_env, get_device
        print("✅ Utility imports successful")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    try:
        import gymnasium as gym
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ External library imports successful")
    except ImportError as e:
        print(f"❌ External library import failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from utils import load_config
        config = load_config("config/config.yaml")
        
        required_keys = ["environment", "training", "algorithms"]
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required config key: {key}")
                return False
        
        print("✅ Configuration loading successful")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_environment_creation():
    """Test environment creation."""
    print("\nTesting environment creation...")
    
    try:
        from utils import create_env
        env = create_env("Pendulum-v1")
        
        # Test environment step
        state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        print("✅ Environment creation and step successful")
        return True
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False


def test_agent_creation():
    """Test agent creation."""
    print("\nTesting agent creation...")
    
    try:
        from agents import DDPGAgent
        from utils import load_config, create_env
        
        config = load_config("config/config.yaml")
        env = create_env("Pendulum-v1")
        
        agent = DDPGAgent(env, config)
        
        # Test prediction
        state, _ = env.reset()
        action, _ = agent.predict(state)
        
        print("✅ Agent creation and prediction successful")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 RL Project Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_loading,
        test_environment_creation,
        test_agent_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python demo.py' to see the project in action")
        print("2. Use 'python cli.py train --help' for CLI training")
        print("3. Use 'streamlit run streamlit_app.py' for the web interface")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
