# simple_evaluate.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import the modified environment
from modified_sumo_env import SumoEnv

# Import the algorithms
try:
    from stable_baselines3 import PPO, A2C
    from sb3_contrib.sac import SAC
except ImportError:
    print("Error importing stable-baselines3 or sb3-contrib.")
    print("Continuing with only the available algorithms...")

def evaluate_model(model_path, algorithm, reward_type, episodes=3):
    """Evaluate a trained model and generate metrics."""
    print(f"Evaluating {model_path}...")
    
    # Create environment with the appropriate reward
    env = SumoEnv(reward_type=reward_type)
    
    # Load model
    if algorithm.lower() == 'ppo':
        model = PPO.load(model_path, env=env)
    elif algorithm.lower() == 'a2c':
        model = A2C.load(model_path, env=env)
    elif algorithm.lower() == 'sac':
        model = SAC.load(model_path, env=env)
    else:
        print(f"Unsupported algorithm: {algorithm}")
        env.close()
        return None
    
    total_rewards = []
    waiting_times = []
    
    # Run evaluation episodes
    for i in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # Handle different gym versions
            obs = obs[0]
            
        done = False
        episode_reward = 0
        episode_waits = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            step_result = env.step(action)
            if len(step_result) == 5:  # New gym version
                obs, reward, done, _, _ = step_result
            else:  # Old gym version
                obs, reward, done, _ = step_result
                
            episode_reward += reward
            
            # Track waiting times
            episode_waits.append(float(obs.sum()))
        
        total_rewards.append(episode_reward)
        waiting_times.append(np.mean(episode_waits))
        
        print(f"Episode {i+1}: Reward={episode_reward:.2f}, Avg Wait={np.mean(episode_waits):.2f}")
    
    # Calculate overall metrics
    avg_reward = np.mean(total_rewards)
    avg_waiting = np.mean(waiting_times)
    
    print(f"Overall: Avg Reward={avg_reward:.2f}, Avg Waiting Time={avg_waiting:.2f}")
    
    # Close environment
    env.close()
    
    return {
        "model": model_path,
        "algorithm": algorithm,
        "reward_type": reward_type,
        "avg_reward": avg_reward,
        "avg_waiting_time": avg_waiting,
        "raw_rewards": total_rewards,
        "raw_waiting_times": waiting_times
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate a SUMO RL model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model file")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["ppo", "a2c", "sac"],
                        help="Algorithm used to train the model")
    parser.add_argument("--reward", type=str, required=True,
                        choices=["simple", "multi_component", "difference"],
                        help="Reward function used")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_model(args.model, args.algorithm, args.reward, args.episodes)
    
    # Save results
    results_file = f"{args.algorithm}_{args.reward}_results.txt"
    with open(results_file, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()