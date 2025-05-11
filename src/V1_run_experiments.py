# run_experiments.py
import os
import time
import subprocess

# Define algorithms and reward functions to test
algorithms = ['ppo', 'sac', 'a2c']
reward_types = ['simple', 'multi_component', 'difference', 'traffic_flow', 'balanced_junction']

# Timesteps for training (reduced for quicker results)
timesteps = 50000

# Results tracking
results = []

# Add the new reward functions
print("Step 1: Updating reward_functions.py with new reward functions...")
# You would need to manually add the new reward functions to reward_functions.py

# Train all combinations
print("\nStep 2: Training all algorithm and reward function combinations...")
for algorithm in algorithms:
    for reward_type in reward_types:
        model_name = f"{algorithm}_{reward_type}"
        print(f"\nTraining {model_name}...")
        
        # Train the model
        subprocess.run([
            'python3', 'train_all_algorithms.py',
            '--algorithm', algorithm,
            '--reward', reward_type,
            '--timesteps', str(timesteps),
            '--name', model_name
        ])
        
        print(f"Completed training {model_name}")
        results.append({
            'algorithm': algorithm,
            'reward_type': reward_type,
            'model_name': model_name
        })

# Evaluate all models
print("\nStep 3: Evaluating all trained models...")
for result in results:
    algorithm = result['algorithm']
    reward_type = result['reward_type']
    model_name = result['model_name']
    
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate the model
    subprocess.run([
        'python3', 'evaluate_models.py',
        '--model', model_name,
        '--algorithm', algorithm,
        '--reward', reward_type,
        '--episodes', '3'  # Reduced for quicker evaluation
    ])
    
    print(f"Completed evaluation of {model_name}")

# Compare all results
print("\nStep 4: Comparing all evaluation results...")
subprocess.run(['python3', 'compare_all_results.py'])

print("\nAll experiments completed! Check the comparison results and graphs.")