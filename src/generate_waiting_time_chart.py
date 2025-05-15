# generate_waiting_time_chart.py
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

def extract_waiting_time_from_tripinfo(filename):
    """Extract average waiting time from a tripinfo XML file."""
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Extract waiting times for all vehicles
        waiting_times = []
        for trip in root.findall('tripinfo'):
            waiting_time = float(trip.get('waitingTime', 0))
            waiting_times.append(waiting_time)
        
        # Calculate average waiting time
        if waiting_times:
            avg_waiting_time = sum(waiting_times) / len(waiting_times)
            print(f"Processed {filename}: Found {len(waiting_times)} vehicles, Avg waiting time: {avg_waiting_time:.2f}s")
            return avg_waiting_time
        else:
            print(f"Warning: No trip data found in {filename}")
            return 0
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return 0

# Define the RL algorithm data
rl_data = [
    ("A2C", "TRAFFIC_FLOW", 11.86),
    ("PPO", "DIFFERENCE", 10.53),
    ("PPO", "SIMPLE", 10.46),
    ("PPO", "TRAFFIC_FLOW", 10.92),
    ("PPO", "MULTI_COMPONENT", 13.67),
    ("A2C", "MULTI_COMPONENT", 12.09),
    ("A2C", "DIFFERENCE", 8.73),
    ("PPO", "BALANCED_JUNCTION", 9.50),
    ("A2C", "SIMPLE", 8.93),
    ("A2C", "BALANCED_JUNCTION", 10.38)
]

# Read baseline data from XML files
print("Reading baseline data from XML files...")
baseline_data = [
    ("BASELINE", "60S", extract_waiting_time_from_tripinfo("baseline_60s_tripinfo.xml")),
    ("BASELINE", "90S", extract_waiting_time_from_tripinfo("baseline_90s_tripinfo.xml")),
    ("BASELINE", "120S", extract_waiting_time_from_tripinfo("baseline_120s_tripinfo.xml"))
]

# Combine all data
all_data = rl_data + baseline_data

# Create labels and values
labels = [f"{algo}\n{reward}" for algo, reward, _ in all_data]
values = [val for _, _, val in all_data]

# Sort data - place RL algorithms first, then baselines
sorted_indices = list(range(len(all_data)))
sorted_indices.sort(key=lambda i: ("Z" if all_data[i][0] == "BASELINE" else all_data[i][0], all_data[i][1]))

# Apply sorting
sorted_labels = [labels[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]

# Create figure
plt.figure(figsize=(15, 8))

# Create the bar chart with different colors for RL vs baseline
colors = ['royalblue' if 'BASELINE' not in label else 'lightcoral' for label in sorted_labels]
bars = plt.bar(range(len(sorted_labels)), sorted_values, color=colors)

# Customize the chart
plt.title('Average Waiting Time Comparison', fontsize=16)
plt.ylabel('Average Waiting Time (s)', fontsize=14)
plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Add a horizontal line for the average of RL methods
rl_avg = np.mean([val for _, _, val in rl_data])
plt.axhline(y=rl_avg, color='green', linestyle='--', 
           label=f'RL Average: {rl_avg:.2f}s')

# Add value labels on bars
for i, v in enumerate(sorted_values):
    plt.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')

# Add legend
plt.legend()

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig('average_waiting_time_comparison.png', dpi=300)
plt.show()

print("Chart generated successfully!")