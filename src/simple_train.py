# simple_train.py
import os
import sys
import time
import argparse

# Import the modified environment
from modified_sumo_env import SumoEnv

# Try importing required packages
try:
    from stable_baselines3 import PPO, A2C
    from sb3_contrib.sac import SAC
except ImportError:
    print("Error importing stable-baselines3 or sb3-contrib.")
    print("Continuing with only the available algorithms...")

def get_algorithm(name, env):
    """Get the algorithm class based on name."""
    if name.lower() == 'ppo':
        return PPO("MlpPolicy", env, verbose=1)
    elif name.lower() == 'a2c':
        return A2C("MlpPolicy", env, verbose=1)
    elif name.lower() == 'sac':
        return SAC("MlpPolicy", env, verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {name}")

def train_model(algorithm='ppo', reward_type='simple', timesteps=20000):
    """Train a model with the specified algorithm and reward type."""
    print(f"Training {algorithm} with {reward_type} reward...")
    
    # Create environment
    env = SumoEnv(reward_type=reward_type)
    
    # Get algorithm
    try:
        model = get_algorithm(algorithm, env)
    except Exception as e:
        print(f"Error creating model: {e}")
        env.close()
        return None
    
    # Train model
    try:
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        return None
    
    # Save model
    model_name = f"{algorithm}_{reward_type}"
    model.save(model_name)
    print(f"Model saved as {model_name}")
    
    # Close environment
    env.close()
    
    return model_name

def main():
    parser = argparse.ArgumentParser(description="Train a SUMO RL model")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "a2c", "sac"],
                       help="RL algorithm to use")
    parser.add_argument("--reward", type=str, default="simple", 
                       choices=["simple", "multi_component", "difference"],
                       help="Reward function to use")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Total timesteps for training")
    
    args = parser.parse_args()
    
    # Train model
    train_model(args.algorithm, args.reward, args.timesteps)

if __name__ == "__main__":
    main()