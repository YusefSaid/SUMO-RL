# train_rl_model.py
import os
import sys
import time
import argparse

# Try to import required libraries
try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO, A2C
    from sb3_contrib import SAC
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages with: pip install stable-baselines3 sb3-contrib gym<=0.21.0")
    sys.exit(1)

# Set up SUMO paths - adjust these to your SUMO installation
SUMO_PATHS = [
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env',
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/tools',
    '/home/yrsaid18/.local/lib/python3.10/site-packages',
    '/usr/share/sumo',
    '/opt/sumo'
]

# Find the first valid SUMO path
for sumo_path in SUMO_PATHS:
    if os.path.exists(sumo_path):
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = sumo_path
            print(f"Set SUMO_HOME to {sumo_path}")
        
        # Add tools directory to path
        tools_path = os.path.join(sumo_path, 'tools')
        if os.path.exists(tools_path) and tools_path not in sys.path:
            sys.path.append(tools_path)
            print(f"Added {tools_path} to Python path")
        break
else:
    print("Warning: Could not find SUMO installation. Import errors may occur.")

# Import SumoEnv and reward functions
try:
    from sumo_env import SumoEnv
    import reward_functions
    print("Successfully imported SUMO environment and reward functions")
except ImportError as e:
    print(f"Error importing SUMO environment: {e}")
    sys.exit(1)

def train_model(algorithm='ppo', reward_type='simple', total_timesteps=20000):
    """Train a model with specified algorithm and reward type"""
    print(f"Starting training with algorithm: {algorithm}, reward type: {reward_type}")
    
    # Create and register the environment
    try:
        gym.envs.register(
            id="CustomSumo-v0",
            entry_point="sumo_env:SumoEnv",
            max_episode_steps=200,
            kwargs={'sumo_cfg': 'sim.sumocfg', 'max_steps': 200, 'reward_type': reward_type}
        )
    except gym.error.Error:
        # Environment might already be registered
        pass
    
    # Create the environment
    try:
        env = gym.make("CustomSumo-v0")
        print("Successfully created SUMO environment")
    except Exception as e:
        print(f"Error creating environment: {e}")
        sys.exit(1)
    
    # Initialize model based on selected algorithm
    try:
        if algorithm.lower() == 'ppo':
            model = PPO("MlpPolicy", env, verbose=1)
        elif algorithm.lower() == 'sac':
            model = SAC("MlpPolicy", env, verbose=1)
        elif algorithm.lower() == 'a2c':
            model = A2C("MlpPolicy", env, verbose=1)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        print(f"Successfully created {algorithm} model")
    except Exception as e:
        print(f"Error creating model: {e}")
        env.close()
        sys.exit(1)
    
    # Train the model
    try:
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        sys.exit(1)
    
    # Save the model
    try:
        model_name = f"{algorithm.lower()}_{reward_type}"
        model.save(model_name)
        print(f"Model saved to {model_name}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Close the environment
    env.close()
    
    return model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SUMO RL model')
    parser.add_argument('--algorithm', type=str, default='ppo',
                      choices=['ppo', 'sac', 'a2c'],
                      help='RL algorithm to use')
    parser.add_argument('--reward', type=str, default='simple',
                      choices=['simple', 'multi_component', 'difference', 'traffic_flow', 'balanced_junction'],
                      help='Reward function to use')
    parser.add_argument('--timesteps', type=int, default=20000,
                      help='Total timesteps for training')
    
    args = parser.parse_args()
    
    # Train the model
    trained_model = train_model(args.algorithm, args.reward, args.timesteps)
    print(f"Trained model: {trained_model}")