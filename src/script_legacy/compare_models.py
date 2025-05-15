# compare_models.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_results(filename):
    """Read results from a file."""
    results = {}
    with open(filename, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip()
                value = value.strip()
                
                # Convert numeric values
                try:
                    if key in ["avg_reward", "avg_waiting_time"]:
                        value = float(value)
                except ValueError:
                    pass
                
                results[key] = value
    
    return results

def compare_all_models():
    """Compare all models based on their evaluation results."""
    # Find all result files
    result_files = glob.glob("*_results.txt")
    
    if not result_files:
        print("No result files found. Run evaluation first.")
        return
    
    # Read all results
    all_results = []
    for file in result_files:
        results = read_results(file)
        all_results.append(results)
    
    # Extract comparison metrics
    algorithms = []
    reward_types = []
    waiting_times = []
    rewards = []
    
    for result in all_results:
        algorithms.append(result.get("algorithm", "unknown"))
        reward_types.append(result.get("reward_type", "unknown"))
        waiting_times.append(result.get("avg_waiting_time", 0))
        rewards.append(result.get("avg_reward", 0))
    
    # Print comparison table
    print("\n=== Model Comparison ===")
    print("Algorithm\tReward Type\tAvg Waiting Time\tAvg Reward")
    for i in range(len(all_results)):
        print(f"{algorithms[i]}\t{reward_types[i]}\t{waiting_times[i]:.2f}\t{rewards[i]:.2f}")
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot waiting times
    labels = [f"{alg}_{rew}" for alg, rew in zip(algorithms, reward_types)]
    ax1.bar(labels, waiting_times)
    ax1.set_title("Average Waiting Time")
    ax1.set_ylabel("Time")
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    
    # Plot rewards
    ax2.bar(labels, rewards)
    ax2.set_title("Average Reward")
    ax2.set_ylabel("Reward")
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("\nComparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    compare_all_models()