# analyze_rewards.py
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Define reward functions to compare
def simple_reward(waiting_times):
    """Original simple reward - negative sum of waiting times"""
    return -np.sum(waiting_times)

def multi_component_reward(waiting_times, prev_waiting_times=None, switched_phase=False):
    """Multi-component reward function"""
    # Waiting time penalty
    waiting_penalty = -0.05 * np.sum(waiting_times)
    
    # Phase switching penalty
    switch_penalty = -0.1 if switched_phase else 0
    
    # Throughput rewards
    free_flow_reward = 0
    cleared_halted_reward = 0
    
    if prev_waiting_times is not None:
        # Count vehicles that might have cleared
        for i in range(len(waiting_times)):
            if waiting_times[i] < prev_waiting_times[i]:
                # Vehicle likely cleared
                free_flow_reward += 0.5
                if prev_waiting_times[i] > 0:
                    # Previously had waiting vehicles
                    cleared_halted_reward += 0.1
    
    return waiting_penalty + switch_penalty + free_flow_reward + cleared_halted_reward

def difference_reward(waiting_times, prev_waiting_times):
    """Reward based on improvement in waiting time"""
    if prev_waiting_times is None:
        return 0
    
    prev_sum = np.sum(prev_waiting_times)
    current_sum = np.sum(waiting_times)
    
    return 0.1 * (prev_sum - current_sum)

# Analyze tripinfo.xml to extract waiting time data
def extract_waiting_times(tripinfo_file):
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        # Extract data
        timesteps = []
        waiting_data = []
        
        # Assume tripinfo has timestep information or can be sorted chronologically
        trips = root.findall("tripinfo")
        
        # Extract waiting times by vehicle
        for trip in trips:
            depart = float(trip.get("depart", 0))
            waiting = float(trip.get("waitingTime", 0))
            timesteps.append(depart)
            waiting_data.append(waiting)
        
        # Sort by departure time
        sorted_data = sorted(zip(timesteps, waiting_data))
        sorted_timesteps, sorted_waiting = zip(*sorted_data)
        
        return np.array(sorted_timesteps), np.array(sorted_waiting)
        
    except Exception as e:
        print(f"Error parsing tripinfo file: {e}")
        return None, None

# Create synthetic simulation data if real data unavailable
def create_synthetic_data(num_steps=200, num_edges=10):
    timesteps = np.arange(num_steps)
    waiting_times = np.zeros((num_steps, num_edges))
    
    # Simulate traffic patterns with peaks at traffic lights
    for t in range(num_steps):
        # Create periodic traffic patterns
        for e in range(num_edges):
            # Base waiting time with some randomness
            waiting_times[t, e] = 5 * np.sin(t/20 + e/3) + 2 * np.random.random()
            waiting_times[t, e] = max(0, waiting_times[t, e])  # No negative waiting times
    
    return timesteps, waiting_times

# Calculate rewards for each function using either real or synthetic data
def compare_reward_functions(waiting_data, timesteps=None):
    num_steps = len(waiting_data)
    results = {
        "simple": np.zeros(num_steps),
        "multi_component": np.zeros(num_steps),
        "difference": np.zeros(num_steps)
    }
    
    # Simulate phase changes every 20 timesteps
    phase_changes = [t % 20 == 0 for t in range(num_steps)]
    
    for t in range(num_steps):
        # Get current and previous waiting times
        current = waiting_data[t]
        prev = waiting_data[t-1] if t > 0 else None
        
        # Calculate rewards for each function
        results["simple"][t] = simple_reward(current)
        results["multi_component"][t] = multi_component_reward(
            current, prev, phase_changes[t])
        if t > 0:
            results["difference"][t] = difference_reward(current, prev)
    
    return results

# Visualize and compare rewards
def visualize_comparison(waiting_data, reward_results, timesteps=None):
    if timesteps is None:
        timesteps = np.arange(len(waiting_data))
    
    # Calculate cumulative rewards
    cumulative_rewards = {
        name: np.cumsum(rewards) 
        for name, rewards in reward_results.items()
    }
    
    # Calculate average waiting times
    avg_waiting = np.mean(waiting_data, axis=1) if waiting_data.ndim > 1 else waiting_data
    
    # Plot waiting times
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(timesteps, avg_waiting, 'b-')
    plt.title('Average Waiting Time')
    plt.xlabel('Timestep')
    plt.ylabel('Waiting Time (s)')
    plt.grid(True)
    
    # Plot instantaneous rewards
    plt.subplot(3, 1, 2)
    for name, rewards in reward_results.items():
        plt.plot(timesteps, rewards, label=name)
    plt.title('Instantaneous Rewards')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative rewards
    plt.subplot(3, 1, 3)
    for name, cum_rewards in cumulative_rewards.items():
        plt.plot(timesteps, cum_rewards, label=name)
    plt.title('Cumulative Rewards')
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_comparison.png')
    print("Visualization saved as 'reward_comparison.png'")
    
    # Calculate summary statistics
    print("\nSummary Statistics:")
    for name, rewards in reward_results.items():
        print(f"{name.capitalize()} Reward:")
        print(f"  Total: {np.sum(rewards):.2f}")
        print(f"  Mean: {np.mean(rewards):.2f}")
        print(f"  Min: {np.min(rewards):.2f}")
        print(f"  Max: {np.max(rewards):.2f}")
    
    # Correlation with waiting time reduction
    waiting_changes = np.diff(avg_waiting)
    waiting_changes = np.append(0, waiting_changes)  # Pad for alignment
    
    print("\nCorrelation with Waiting Time Reduction:")
    for name, rewards in reward_results.items():
        corr = np.corrcoef(rewards, -waiting_changes)[0, 1]
        print(f"  {name.capitalize()}: {corr:.4f}")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze different reward functions')
    parser.add_argument('--tripinfo', type=str, default=None,
                       help='Path to tripinfo.xml file')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of parsing tripinfo')
    
    args = parser.parse_args()
    
    if args.tripinfo and not args.synthetic:
        # Use real data from tripinfo.xml
        timesteps, waiting_data = extract_waiting_times(args.tripinfo)
        if timesteps is None:
            print("Failed to parse tripinfo file. Using synthetic data instead.")
            args.synthetic = True
    
    if args.synthetic or timesteps is None:
        # Use synthetic data for testing
        print("Generating synthetic traffic data...")
        timesteps, waiting_data = create_synthetic_data()
    
    # Compare reward functions
    reward_results = compare_reward_functions(waiting_data, timesteps)
    
    # Visualize comparison
    visualize_comparison(waiting_data, reward_results, timesteps)