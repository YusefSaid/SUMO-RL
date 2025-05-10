import os
import time
from stable_baselines3 import PPO, A2C
from sb3_contrib import SAC
from sumo_env import SumoEnv

# Define algorithms and reward functions to test
algorithms = {
    'ppo': PPO,
    'a2c': A2C,
    'sac': SAC
}

reward_functions = [
    'simple',
    'multi_component',
    'difference',
    'traffic_density',
    'velocity_based',
    'queue_length',
    'time_loss',
    'adaptive_weighted',
    'emergency_vehicle',
    'delay_throughput',
    'emission_reduction'
]

# Training settings
timesteps = 5000  # Short training for quick testing
episodes_to_evaluate = 5

# Test loop
for algo_name, algo_class in algorithms.items():
    for reward_type in reward_functions:
        print(f"\n=== Training {algo_name} with {reward_type} reward ===")
        
        # Create environment
        env = SumoEnv(sumo_cfg="sim.sumocfg", max_steps=200, reward_type=reward_type)
        
        # Create and train model
        model = algo_class("MlpPolicy", env, verbose=0)
        
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Quick evaluation
        total_rewards = []
        total_waiting = []
        
        for i in range(episodes_to_evaluate):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_waiting = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_waiting.append(float(obs.sum()))
            
            avg_waiting = sum(episode_waiting) / len(episode_waiting) if episode_waiting else 0
            total_rewards.append(episode_reward)
            total_waiting.append(avg_waiting)
            
            print(f"Episode {i+1}: Reward={episode_reward:.2f}, Avg Waiting={avg_waiting:.2f}")
        
        # Calculate average metrics
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_waiting = sum(total_waiting) / len(total_waiting)
        
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Waiting Time: {avg_waiting:.2f}")
        
        # Save results to file
        with open(f"{algo_name}_{reward_type}_results.txt", "w") as f:
            f.write(f"algorithm: {algo_name}\n")
            f.write(f"reward_type: {reward_type}\n")
            f.write(f"avg_reward: {avg_reward:.6f}\n")
            f.write(f"avg_waiting_time: {avg_waiting:.6f}\n")
        
        # Close environment
        env.close()