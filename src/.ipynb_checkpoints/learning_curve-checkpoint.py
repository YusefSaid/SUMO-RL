# learning_curve.py
import matplotlib.pyplot as plt
import numpy as np

# Sample learning data - replace with your actual data if available
episodes = np.arange(0, 100, 5)
a2c_diff_rewards = 20 * np.exp(-0.03 * episodes) + 8 + np.random.normal(0, 1, len(episodes))
a2c_simple_rewards = 22 * np.exp(-0.025 * episodes) + 9 + np.random.normal(0, 1.2, len(episodes))

plt.figure(figsize=(10, 6))
plt.plot(episodes, a2c_diff_rewards, 'o-', label='A2C with Difference Reward', color='royalblue')
plt.plot(episodes, a2c_simple_rewards, 's-', label='A2C with Simple Reward', color='darkorange')

plt.title('Learning Progress During Training', fontsize=16)
plt.xlabel('Training Episode', fontsize=14)
plt.ylabel('Average Waiting Time (s)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('learning_progress.png', dpi=300)