#!/usr/bin/env python3
"""Setup script for the RL Robotic Control project."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = ["logs", "checkpoints", "notebooks"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run the test suite."""
    print("\n🧪 Running tests...")
    return run_command("python test_setup.py", "Running test suite")


def main():
    """Main setup function."""
    print("🚀 RL Robotic Control Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now:")
    print("1. Run 'python demo.py' to see the project in action")
    print("2. Use 'python cli.py train --help' for CLI training")
    print("3. Use 'streamlit run streamlit_app.py' for the web interface")
    print("4. Check the README.md for detailed usage instructions")


if __name__ == "__main__":
    main()
