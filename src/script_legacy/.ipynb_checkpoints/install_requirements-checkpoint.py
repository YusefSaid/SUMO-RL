# install_requirements.py
import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
packages = [
    "stable-baselines3",
    "sb3-contrib",  # Contains SAC for discrete action spaces
    "matplotlib",
    "numpy",
    "gym<=0.21.0"   # Use older gym version for compatibility
]

for package in packages:
    try:
        install_package(package)
    except Exception as e:
        print(f"Failed to install {package}: {e}")

print("All required packages installed successfully!")