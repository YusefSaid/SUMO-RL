# generate_time_loss_chart.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use your actual data from the experiments
# Based on the values from your Image 4 evaluation results
data = {
    'algorithm': ['A2C', 'A2C', 'PPO', 'PPO', 'PPO', 'A2C'],
    'reward_type': ['DIFFERENCE', 'SIMPLE', 'BALANCED_JUNCTION', 'DIFFERENCE', 'SIMPLE', 'TRAFFIC_FLOW'],
    'loss_percentage': [63.45, 64.00, 64.57, 66.79, 64.79, 69.20]
}

df = pd.DataFrame(data)

# Sort by loss percentage (best performing first)
df = df.sort_values('loss_percentage')

# Create a bar chart for time loss percentages
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(df)), df['loss_percentage'], color='salmon')

# Add algorithm and reward type labels
labels = [f"{algo}\n{reward}" for algo, reward in zip(df['algorithm'], df['reward_type'])]
plt.xticks(range(len(df)), labels, rotation=0)

# Add value labels on top of bars
for i, v in enumerate(df['loss_percentage']):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')

plt.title('Time Loss Percentage by Algorithm and Reward Function', fontsize=16)
plt.ylabel('Time Loss (%)', fontsize=14)
plt.ylim(0, max(df['loss_percentage']) * 1.15)  # Add some headroom for labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('time_loss_comparison.png', dpi=300)
plt.show()