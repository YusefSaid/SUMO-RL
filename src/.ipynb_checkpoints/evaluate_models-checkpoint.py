import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from sb3_contrib import SAC
import gym
from gym.envs.registration import register

# Import your environment
from sumo_env import SumoEnv

def run_evaluation(model_path, algorithm, reward_type, episodes=5):
    """Run evaluation for a single model"""
    print(f"\nEvaluating {model_path} (Algorithm: {algorithm}, Reward: {reward_type})")
    
    # Register environment if needed
    try:
        register(
            id="CustomSumo-v0",
            entry_point="sumo_env:SumoEnv",
            max_episode_steps=200,
        )
    except:
        pass
    
    # Create environment with the same reward type
    env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=200, reward_type=reward_type)
    
    # Load the appropriate model
    if algorithm.lower() == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algorithm.lower() == 'a2c':
        model = A2C.load(model_path, env=env)
    elif algorithm.lower() == 'sac':
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Run the evaluation episodes
    for i in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"Episode {i+1}: Reward={total_reward:.2f}, Steps={steps}")
    
    # Generate tripinfo.xml from the last run
    output_filename = f"{algorithm}_{reward_type}_tripinfo.xml"
    
    # Rename the generated tripinfo.xml to our custom name
    try:
        os.rename("tripinfo.xml", output_filename)
        print(f"Generated evaluation output: {output_filename}")
    except:
        print(f"Failed to rename tripinfo.xml to {output_filename}")
    
    # Close the environment
    env.close()
    
    return output_filename

def analyze_results(tripinfo_file):
    """Run the analyze_results.py script on a tripinfo file"""
    output = subprocess.check_output(['python3', 'analyze_results.py', tripinfo_file])
    return output.decode('utf-8')

def compare_all_models(models_info):
    """Compare all trained models"""
    results = {}
    
    for info in models_info:
        output_file = run_evaluation(
            info['path'], 
            info['algorithm'], 
            info['reward_type']
        )
        
        # Store the output file for analysis
        info['output_file'] = output_file
        
        # Extract metrics using your existing analysis script
        # This is simplified; you would parse the output of analyze_results.py
        # For now, we'll just collect the files for later analysis
        results[f"{info['algorithm']}_{info['reward_type']}"] = output_file
    
    # Now use your compare_all_results.py script to analyze all files
    print("\nGenerating comparative analysis...")
    subprocess.run(['python3', 'compare_all_results.py'])
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RL models for SUMO traffic control')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model file to evaluate')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['ppo', 'sac', 'a2c'],
                        help='Algorithm used to train the model')
    parser.add_argument('--reward', type=str, required=True,
                        help='Reward type used for training')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    
    args = parser.parse_args()
    
    # Run evaluation for the specified model
    output_file = run_evaluation(
        args.model, 
        args.algorithm, 
        args.reward, 
        args.episodes
    )
    
    print(f"\nEvaluation completed. Results saved to {output_file}")