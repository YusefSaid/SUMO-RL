# generate_reward_impact.py - simplified version
import matplotlib.pyplot as plt
import numpy as np

# Explicitly define the data - no dependencies on external variables
a2c_data = {
    "DIFFERENCE": 8.73,
    "SIMPLE": 8.93,
    "BALANCED_JUNCTION": 10.38,
    "TRAFFIC_FLOW": 11.86,
    "MULTI_COMPONENT": 12.09
}

ppo_data = {
    "BALANCED_JUNCTION": 9.50,
    "SIMPLE": 10.46,
    "DIFFERENCE": 10.53,
    "TRAFFIC_FLOW": 10.92,
    "MULTI_COMPONENT": 13.67
}

# Sort the data
a2c_labels = sorted(a2c_data.keys(), key=lambda k: a2c_data[k])
a2c_values = [a2c_data[k] for k in a2c_labels]

ppo_labels = sorted(ppo_data.keys(), key=lambda k: ppo_data[k])
ppo_values = [ppo_data[k] for k in ppo_labels]

# Create A2C plot
plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(a2c_labels)), a2c_values, color='royalblue')

# Add labels
plt.xticks(range(len(a2c_labels)), a2c_labels)
plt.title('Impact of Reward Functions on A2C Performance', fontsize=16)
plt.ylabel('Average Waiting Time (seconds)', fontsize=14)
plt.xlabel('Reward Function Type', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(a2c_values):
    plt.text(i, v + 0.2, f"{v:.2f}s", ha='center', fontweight='bold')

# Add average line
avg = sum(a2c_values) / len(a2c_values)
plt.axhline(y=avg, color='red', linestyle='--', 
           label=f'Average: {avg:.2f}s')
plt.legend()

plt.tight_layout()
plt.savefig('a2c_reward_impact.png', dpi=300)

# Create PPO plot
plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(ppo_labels)), ppo_values, color='firebrick')

# Add labels
plt.xticks(range(len(ppo_labels)), ppo_labels)
plt.title('Impact of Reward Functions on PPO Performance', fontsize=16)
plt.ylabel('Average Waiting Time (seconds)', fontsize=14)
plt.xlabel('Reward Function Type', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for i, v in enumerate(ppo_values):
    plt.text(i, v + 0.2, f"{v:.2f}s", ha='center', fontweight='bold')

# Add average line
avg = sum(ppo_values) / len(ppo_values)
plt.axhline(y=avg, color='blue', linestyle='--', 
           label=f'Average: {avg:.2f}s')
plt.legend()

plt.tight_layout()
plt.savefig('ppo_reward_impact.png', dpi=300)

# Create combined plot
plt.figure(figsize=(14, 8))

# Get combined labels (all unique reward functions)
all_labels = sorted(set(a2c_labels).union(set(ppo_labels)))
x = np.arange(len(all_labels))
width = 0.35

# Prepare data for the combined plot
a2c_combined = [a2c_data.get(label, 0) for label in all_labels]
ppo_combined = [ppo_data.get(label, 0) for label in all_labels]

# Create the grouped bars
plt.bar(x - width/2, a2c_combined, width, label='A2C', color='royalblue')
plt.bar(x + width/2, ppo_combined, width, label='PPO', color='firebrick')

# Add labels and title
plt.xlabel('Reward Function Type', fontsize=14)
plt.ylabel('Average Waiting Time (seconds)', fontsize=14)
plt.title('Comparison of Reward Functions Across Algorithms', fontsize=16)
plt.xticks(x, all_labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('combined_reward_impact.png', dpi=300)

print("All visualizations have been successfully generated!")