# compare_results.py
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

def compare_all_results():
    """Compare all results files."""
    # Find all result files
    result_files = glob.glob("*_results.txt")
    
    if not result_files:
        print("No result files found. Run evaluation first.")
        return
    
    # Read and process results
    all_results = []
    for file in result_files:
        results = read_results(file)
        all_results.append(results)
    
    # Sort results by algorithm and reward type
    all_results.sort(key=lambda x: (x.get("algorithm", ""), x.get("reward_type", "")))
    
    # Print comparison table
    print("\n===== Comparison of All Models =====")
    print("Algorithm\tReward Type\tAvg Waiting Time\tAvg Reward")
    print("-" * 70)
    
    for result in all_results:
        alg = result.get("algorithm", "unknown")
        reward = result.get("reward_type", "unknown")
        waiting = float(result.get("avg_waiting_time", 0))
        reward_val = float(result.get("avg_reward", 0))
        print(f"{alg}\t\t{reward}\t\t{waiting:.2f}\t\t{reward_val:.2f}")
    
    # Create visualization
    # Extract data for plotting
    algos = [r.get("algorithm", "unknown") for r in all_results]
    reward_types = [r.get("reward_type", "unknown") for r in all_results]
    waiting_times = [float(r.get("avg_waiting_time", 0)) for r in all_results]
    reward_vals = [float(r.get("avg_reward", 0)) for r in all_results]
    
    # Create labels
    labels = [f"{a}_{r}" for a, r in zip(algos, reward_types)]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot waiting times
    bars1 = ax1.bar(labels, waiting_times)
    ax1.set_title("Average Waiting Time by Model")
    ax1.set_ylabel("Waiting Time")
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    
    # Plot rewards
    bars2 = ax2.bar(labels, reward_vals)
    ax2.set_title("Average Reward by Model")
    ax2.set_ylabel("Reward")
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    print("\nComparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    compare_all_results()