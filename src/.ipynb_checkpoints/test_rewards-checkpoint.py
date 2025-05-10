# test_rewards.py

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from reward_functions import simple_waiting_reward, multi_component_reward, difference_reward

# Test environment with fixed actions to evaluate reward functions
def test_reward_function(reward_type, num_episodes=10, max_steps=200):
    from gym.envs.registration import register

    try:
        register(
            id="CustomSumo-v0",
            entry_point="sumo_env:SumoEnv",
            max_episode_steps=max_steps,
        )
    except:
        pass

    # Create environment with the specified reward type
    env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=max_steps, reward_type=reward_type)
    
    all_rewards = []
    all_waiting_times = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        waiting_times = []
        
        for step in range(max_steps):
            # Take a simple action (just cycle through available phases)
            action = step % env.action_space.n
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Record waiting time
            waiting_times.append(float(np.sum(next_state)))
            
            if done:
                break
                
            state = next_state
        
        all_rewards.append(episode_reward)
        all_waiting_times.append(np.mean(waiting_times))
        
        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {episode_reward:.2f}, " 
              f"Avg Waiting Time: {np.mean(waiting_times):.2f}")
    
    env.close()
    
    # Summarize results
    avg_reward = np.mean(all_rewards)
    avg_waiting_time = np.mean(all_waiting_times)
    
    print(f"\n--- Results for {reward_type} Reward ---")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Waiting Time: {avg_waiting_time:.2f}")
    
    return {
        'reward_type': reward_type,
        'avg_reward': avg_reward,
        'avg_waiting_time': avg_waiting_time,
        'all_rewards': all_rewards,
        'all_waiting_times': all_waiting_times
    }

def run_all_reward_tests():
    results = {}
    
    for reward_type in ['simple', 'multi_component', 'difference']:
        print(f"\nTesting {reward_type} reward function...")
        results[reward_type] = test_reward_function(reward_type)
    
    # Plot comparison
    reward_types = list(results.keys())
    avg_rewards = [results[r]['avg_reward'] for r in reward_types]
    avg_waiting_times = [results[r]['avg_waiting_time'] for r in reward_types]
    
    # Create bar charts
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(reward_types, avg_rewards)
    plt.title('Average Reward by Reward Type')
    plt.ylabel('Average Reward')
    
    plt.subplot(1, 2, 2)
    plt.bar(reward_types, avg_waiting_times)
    plt.title('Average Waiting Time by Reward Type')
    plt.ylabel('Average Waiting Time (s)')
    
    plt.tight_layout()
    plt.savefig('reward_function_comparison.png')
    
    print("\nResults saved to reward_function_comparison.png")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test different reward functions')
    parser.add_argument('--reward', type=str, default=None, 
                        choices=['simple', 'multi_component', 'difference'],
                        help='Specific reward function to test (default: test all)')
    
    args = parser.parse_args()
    
    if args.reward:
        test_reward_function(args.reward)
    else:
        run_all_reward_tests()