# minimal_train.py
import os
import gym
import numpy as np
from stable_baselines3 import PPO

# Disable TensorFlow warnings and TensorBoard dependency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your environment and reward functions
from sumo_env import SumoEnv
import reward_functions  # This ensures they're available for the environment

# Register the environment
from gym.envs.registration import register

# Function to train a model with a specific reward type
def train_model(reward_type='simple', total_timesteps=50000):
    # Create and register the environment
    try:
        register(
            id="CustomSumo-v0",
            entry_point="sumo_env:SumoEnv",
            max_episode_steps=200,
        )
    except:
        # Environment might already be registered
        pass
    
    # Create the environment
    env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=200, reward_type=reward_type)
    
    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, 
                # Explicitly disable tensorboard
                tensorboard_log=None)
    
    print(f"Starting training with reward type: {reward_type}")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model_path = f"ppo_{reward_type}"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SUMO RL agent with minimal dependencies')
    parser.add_argument('--reward', type=str, default='simple', 
                        choices=['simple', 'multi_component', 'difference'],
                        help='Reward function to use')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Total timesteps for training')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args.reward, args.timesteps)