import os
import sys
import subprocess

# Check if we're in the correct virtualenv
if not hasattr(sys, "real_prefix") and not sys.base_prefix != sys.prefix:
    print("WARNING: You might not be in your sumo_env virtualenv!")

# Set SUMO_HOME correctly
# Adjust this path to where SUMO is actually installed
os.environ["SUMO_HOME"] = "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo"

# Configure SUMO to run without GUI since we're missing X11 libraries
os.environ["SUMO_GUI_ENABLED"] = "false"  # Disable GUI
os.environ["OSG_GL"] = "OSMESA"  # Use software rendering

# For SUMO-RL to work without X11 dependencies, force it to use libsumo instead of traci
os.environ["LIBSUMO_AS_TRACI"] = "1"

print("SUMO environment configured to run headless (no GUI)")

# Import stable_baselines3
try:
    from stable_baselines3 import PPO, A2C
    try:
        from sb3_contrib import SAC
    except ImportError:
        SAC = None
        print("WARNING: sb3_contrib.SAC not installed. SAC algorithm won't be available.")
except ImportError:
    print("ERROR: stable_baselines3 not installed. Please install it with: pip install stable_baselines3")
    sys.exit(1)

def run_evaluation(model_path, algorithm, reward_type, episodes=5):
    print(f"Evaluating {algorithm} model with {reward_type} reward for {episodes} episodes")
    
    # Initialize environment here
    # env = ...
    
    # Build the actual filename SB3 expects
    fname = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(fname):
        print(f"  ➜  Model file {fname} not found, skipping.")
        return None

    # Display environment info before loading
    print(f"Environment information:")
    print(f"  SUMO_HOME: {os.environ.get('SUMO_HOME', 'Not set')}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Load the model
    alg = algorithm.lower()
    try:
        if alg == 'ppo':
            model = PPO.load(fname, env=env)
        elif alg == 'a2c':
            model = A2C.load(fname, env=env)
        elif alg == 'sac':
            if SAC is None:
                raise RuntimeError("sb3_contrib.SAC not installed in this env!")
            model = SAC.load(fname, env=env)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        print(f"Successfully loaded {alg} model from {fname}")
        
        # Evaluation code would go here
        # ...
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None