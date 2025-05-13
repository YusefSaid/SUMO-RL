# generate_comparison_chart.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from your evaluations (from Image 4)
algorithms = ["A2C", "A2C", "A2C", "PPO", "PPO", "PPO"]
reward_types = ["DIFFERENCE", "SIMPLE", "BALANCED_JUNCTION", 
                "DIFFERENCE", "SIMPLE", "BALANCED_JUNCTION"]
waiting_times = [8.73, 8.93, 10.38, 10.53, 10.46, 9.50]
time_losses = [15.57, 16.06, 18.24, 18.21, 18.04, 17.11]
loss_percentages = [63.45, 64.00, 71.86, 66.79, 64.79, 64.57]

# Create DataFrame
df = pd.DataFrame({
    'algorithm': algorithms,
    'reward_type': reward_types,
    'waiting_time': waiting_times,
    'time_loss': time_losses,
    'loss_percentage': loss_percentages
})

# Sort by waiting time (best first)
df = df.sort_values('waiting_time')

# Create a bar chart for waiting times
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(df)), df['waiting_time'], color='skyblue')

# Add algorithm and reward type labels
labels = [f"{algo}\n{reward}" for algo, reward in zip(df['algorithm'], df['reward_type'])]
plt.xticks(range(len(df)), labels)

# Add value labels on top of bars
for i, v in enumerate(df['waiting_time']):
    plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

plt.title('Average Waiting Time by Algorithm and Reward Function')
plt.ylabel('Waiting Time (seconds)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add a horizontal line for the average
plt.axhline(y=np.mean(df['waiting_time']), color='red', linestyle='--', 
           label=f'Average: {np.mean(df["waiting_time"]):.2f}s')
plt.legend()

plt.tight_layout()
plt.savefig('algorithm_reward_comparison.png', dpi=300)
plt.show()