# improvement_chart.py
import matplotlib.pyplot as plt
import numpy as np

# Define data - use your actual values
baseline_60s = 22.15
baseline_90s = 15.03
baseline_120s = 29.78
baseline_avg = (baseline_60s + baseline_90s + baseline_120s) / 3
top_performers = [
    ("A2C\nDIFFERENCE", 8.73),
    ("A2C\nSIMPLE", 8.93),
    ("PPO\nBALANCED", 9.50)
]

# Calculate improvement percentages
improvements = [(name, 100 * (baseline_avg - value) / baseline_avg) 
                for name, value in top_performers]

# Create plot
plt.figure(figsize=(10, 6))
# Replace 'bronze' with an RGB value for bronze-like color
bars = plt.bar([x[0] for x in improvements], [x[1] for x in improvements], 
              color=['gold', 'silver', '#CD7F32'])  # #CD7F32 is bronze color

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.title('Waiting Time Reduction vs. Baseline Average', fontsize=16)
plt.ylabel('Improvement Percentage (%)', fontsize=14)
plt.ylim(0, max([x[1] for x in improvements]) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('improvement_chart.png', dpi=300)