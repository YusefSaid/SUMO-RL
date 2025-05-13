# generate_complete_time_loss_chart.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Complete data from your Image 4 evaluation results
data = [
    ("A2C", "TRAFFIC_FLOW", 11.86, 19.38, 69.20),
    ("PPO", "DIFFERENCE", 10.53, 18.21, 66.79),
    ("PPO", "SIMPLE", 10.46, 18.04, 64.79),
    ("PPO", "TRAFFIC_FLOW", 10.92, 18.39, 71.20),
    ("PPO", "MULTI_COMPONENT", 13.67, 21.00, 70.14),
    ("A2C", "MULTI_COMPONENT", 12.09, 20.07, 76.93),
    ("A2C", "DIFFERENCE", 8.73, 15.57, 63.45),
    ("PPO", "BALANCED_JUNCTION", 9.50, 17.11, 64.57),
    ("A2C", "SIMPLE", 8.93, 16.06, 64.00),
    ("A2C", "BALANCED_JUNCTION", 10.38, 18.24, 71.86)
]

# Convert to DataFrame
columns = ["algorithm", "reward_type", "avg_waiting", "avg_loss", "loss_percentage"]
df = pd.DataFrame(data, columns=columns)

# Sort by loss percentage (best performing first)
df = df.sort_values('loss_percentage')

# Create a bar chart for time loss percentages
plt.figure(figsize=(15, 8))

# Create a colormap to distinguish algorithms
colors = {'A2C': 'salmon', 'PPO': 'lightblue'}
bar_colors = [colors[algo] for algo in df['algorithm']]

bars = plt.bar(range(len(df)), df['loss_percentage'], color=bar_colors)

# Add algorithm and reward type labels
labels = [f"{algo}\n{reward}" for algo, reward in zip(df['algorithm'], df['reward_type'])]
plt.xticks(range(len(df)), labels, rotation=45, ha='right')

# Add value labels on top of bars
for i, v in enumerate(df['loss_percentage']):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')

plt.title('Time Loss Percentage by Algorithm and Reward Function', fontsize=16)
plt.ylabel('Time Loss (%)', fontsize=14)
plt.ylim(0, max(df['loss_percentage']) * 1.15)  # Add some headroom for labels
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=algo) for algo, color in colors.items()]
plt.legend(handles=legend_elements, title="Algorithms")

plt.tight_layout()
plt.savefig('time_loss_comparison.png', dpi=300)
plt.show()