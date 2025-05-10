# quick_train.py
import os
import sys
import time
import argparse
import numpy as np

# Import the mock environment
from mock_sumo_env import MockSumoEnv

try:
    from stable_baselines3 import PPO, A2C
    has_sb3 = True
except ImportError:
    print("Warning: stable-baselines3 not found. Only PPO will be available.")
    has_sb3 = False

def train_model(algorithm='ppo', reward_type='simple', timesteps=20000):
    """Train a model with the specified algorithm and reward type."""
    print(f"Training {algorithm} with {reward_type} reward for {timesteps} steps...")
    
    # Create environment
    env = MockSumoEnv(reward_type=reward_type, max_steps=200)
    print("Created mock SUMO environment")
    
    # Create model
    if algorithm.lower() == 'ppo':
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
        print("Created PPO model")
    elif algorithm.lower() == 'a2c' and has_sb3:
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, verbose=1)
        print("Created A2C model")
    else:
        print(f"Unsupported algorithm: {algorithm}")
        env.close()
        return None
    
    # Train model
    start_time = time.time()
    print("Starting training...")
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    # Save model
    model_name = f"{algorithm}_{reward_type}"
    model.save(model_name)
    print(f"Model saved as {model_name}")
    
    # Close environment
    env.close()
    
    return model_name

def evaluate_model(model_path, algorithm, reward_type, episodes=10):
    """Evaluate a trained model."""
    print(f"Evaluating {model_path}...")
    
    # Create environment
    env = MockSumoEnv(reward_type=reward_type, max_steps=200)
    
    # Load model
    if algorithm.lower() == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
    elif algorithm.lower() == 'a2c':
        from stable_baselines3 import A2C
        model = A2C.load(model_path, env=env)
    else:
        print(f"Unsupported algorithm: {algorithm}")
        env.close()
        return None
    
    # Evaluate model
    all_waiting_times = []
    all_rewards = []
    
    for i in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_waiting_times = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_waiting_times.append(float(obs.sum()))
        
        avg_waiting = np.mean(episode_waiting_times)
        all_waiting_times.append(avg_waiting)
        all_rewards.append(episode_reward)
        
        print(f"Episode {i+1}: Reward={episode_reward:.2f}, Avg Waiting={avg_waiting:.2f}")
    
    # Calculate overall metrics
    avg_reward = np.mean(all_rewards)
    avg_waiting = np.mean(all_waiting_times)
    
    print(f"Overall metrics for {algorithm} with {reward_type} reward:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Waiting Time: {avg_waiting:.2f}")
    
    # Save results
    results = {
        "algorithm": algorithm,
        "reward_type": reward_type,
        "avg_reward": avg_reward,
        "avg_waiting_time": avg_waiting,
    }
    
    # Save to file
    with open(f"{algorithm}_{reward_type}_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    env.close()
    return results

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RL models for traffic control")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "a2c"],
                       help="RL algorithm to use")
    parser.add_argument("--reward", type=str, default="simple", 
                       choices=["simple", "multi_component", "difference"],
                       help="Reward function to use")
    parser.add_argument("--timesteps", type=int, default=20000,
                       help="Total timesteps for training")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate after training")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    # Train model
    model_name = train_model(args.algorithm, args.reward, args.timesteps)
    
    # Evaluate if requested
    if args.evaluate and model_name:
        evaluate_model(model_name, args.algorithm, args.reward, args.episodes)

if __name__ == "__main__":
    main()