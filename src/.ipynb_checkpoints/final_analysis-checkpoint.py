# final_analysis.py
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Hard-code your results from Image 4 for consistency
results_data = [
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

# Create DataFrame
columns = ["algorithm", "reward_type", "avg_waiting", "avg_loss", "loss_percentage"]
df = pd.DataFrame(results_data, columns=columns)

# Sort by waiting time (best first)
df = df.sort_values("avg_waiting")

# 1. Create bar chart for average waiting time
plt.figure(figsize=(14, 8))
ax = sns.barplot(x="reward_type", y="avg_waiting", hue="algorithm", data=df, palette="Set2")

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f')

plt.title('Average Waiting Time by Algorithm and Reward Function', fontsize=16)
plt.xlabel('Reward Function', fontsize=14)
plt.ylabel('Average Waiting Time (seconds)', fontsize=14)
plt.legend(title='Algorithm')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('final_waiting_time_comparison.png', dpi=300)

# 2. Create heatmap for all metrics
plt.figure(figsize=(16, 10))
heatmap_data = df.pivot_table(
    index='algorithm', 
    columns='reward_type', 
    values='avg_waiting'
)
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
plt.title('Average Waiting Time Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('waiting_time_heatmap.png', dpi=300)

# 3. Create summary chart of best performers
top_performers = df.head(3)  # Top 3 performers
plt.figure(figsize=(10, 6))
bars = plt.bar(
    [f"{row.algorithm} {row.reward_type}" for _, row in top_performers.iterrows()],
    top_performers['avg_waiting'],
    color=['gold', 'silver', 'sandybrown']
)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.2f}s', ha='center', va='bottom')

plt.title('Top 3 Performing Models', fontsize=16)
plt.ylabel('Average Waiting Time (seconds)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('top_performers.png', dpi=300)

# 4. Create multi-metric comparison for all models
plt.figure(figsize=(14, 10))

# Create positions for the bars
x = np.arange(len(df))
width = 0.3

# Create grouped bars
plt.bar(x - width, df['avg_waiting'], width, label='Avg Waiting Time (s)', color='royalblue')
plt.bar(x, df['avg_loss'], width, label='Avg Time Loss (s)', color='firebrick')
plt.bar(x + width, df['loss_percentage'], width, label='Loss Percentage (%)', color='forestgreen')

# Add labels and title
labels = [f"{row.algorithm}\n{row.reward_type}" for _, row in df.iterrows()]
plt.xticks(x, labels, rotation=45, ha='right')
plt.title('Comprehensive Performance Comparison', fontsize=16)
plt.ylabel('Value', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_comparison.png', dpi=300)

print("Analysis complete! Four comparison charts have been generated.")