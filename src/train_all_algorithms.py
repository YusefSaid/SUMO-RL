import os
import gym
import numpy as np
from stable_baselines3 import PPO, A2C
try:
    from sb3_contrib import SAC
except ImportError:
    SAC = None
  # SAC for discrete actions is in sb3_contrib

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import your environment and reward functions
from sumo_env import SumoEnv
import reward_functions

# Register the environment
from gym.envs.registration import register

def train_model(algorithm='ppo', reward_type='simple', total_timesteps=20000, model_name=None):
    """Train a model with specified algorithm and reward type"""
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
    
    # Initialize model based on selected algorithm
    if algorithm.lower() == 'ppo':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    elif algorithm.lower() == 'sac':
        # Note: For discrete action spaces, we use SAC from sb3_contrib
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=None)
    elif algorithm.lower() == 'a2c':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=None)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print(f"Starting training with algorithm: {algorithm}, reward type: {reward_type}")
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    if model_name is None:
        model_name = f"{algorithm.lower()}_{reward_type}"
    model.save(model_name)
    print(f"Model saved to {model_name}")
    
    # Close the environment
    env.close()
    
    return model_name

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SUMO RL agent with various algorithms')
    parser.add_argument('--algorithm', type=str, default='ppo', 
                        choices=['ppo', 'sac', 'a2c'],
                        help='RL algorithm to use')
    parser.add_argument('--reward', type=str, default='simple', 
                        choices=['simple', 'multi_component', 'difference', 'traffic_flow', 'balanced_junction'],
                        help='Reward function to use')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Total timesteps for training')
    parser.add_argument('--name', type=str, default=None,
                        help='Custom name for the saved model')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args.algorithm, args.reward, args.timesteps, args.name)