# generate_metrics_comparison.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use your actual data from the experiments
# Based on the values from your Image 4 evaluation results
data = [
    ("A2C", "DIFFERENCE", 8.73, 15.57, 63.45),
    ("A2C", "SIMPLE", 8.93, 16.06, 64.00),
    ("PPO", "BALANCED_JUNCTION", 9.50, 17.11, 64.57),
    ("PPO", "SIMPLE", 10.46, 18.04, 64.79),
    ("PPO", "DIFFERENCE", 10.53, 18.21, 66.79),
    ("A2C", "TRAFFIC_FLOW", 11.86, 19.38, 69.20)
]

# Convert to DataFrame - using your top 6 performers for clarity
columns = ["algorithm", "reward_type", "waiting_time", "time_loss", "loss_percentage"]
df = pd.DataFrame(data, columns=columns)

# Create a grouped bar chart for multiple metrics
fig, ax = plt.subplots(figsize=(14, 8))

# Set the width of each bar and positions
width = 0.35
x = np.arange(len(df))

# Create bars
bars1 = ax.bar(x - width/2, df['waiting_time'], width, label='Avg Waiting Time (s)', color='skyblue')
bars2 = ax.bar(x + width/2, df['time_loss'], width, label='Avg Time Loss (s)', color='lightcoral')

# Add algorithm and reward type labels
labels = [f"{algo}\n{reward}" for algo, reward in zip(df['algorithm'], df['reward_type'])]
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Add a second y-axis for loss percentage
ax2 = ax.twinx()
ax2.plot(x, df['loss_percentage'], 'o-', color='darkgreen', label='Loss Percentage (%)')
ax2.set_ylabel('Loss Percentage (%)', fontsize=12)

# Add value labels on bars
for i, v in enumerate(df['waiting_time']):
    ax.text(i - width/2, v + 0.3, f"{v:.1f}", ha='center', va='bottom', fontsize=9)
for i, v in enumerate(df['time_loss']):
    ax.text(i + width/2, v + 0.3, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

# Add title and labels
ax.set_title('Performance Metrics by Algorithm and Reward Function', fontsize=16)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_ylim(0, max(max(df['waiting_time']), max(df['time_loss'])) * 1.15)  # Add headroom for labels
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add legends
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('performance_metrics_comparison.png', dpi=300)
plt.show()