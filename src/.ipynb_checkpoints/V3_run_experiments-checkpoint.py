# V3_run_experiments.py
import os
import time
import subprocess

# Define algorithms and reward functions to test - SAC removed
algorithms = ['ppo', 'a2c']  # Removed 'sac'
reward_types = ['simple', 'multi_component', 'difference', 'traffic_flow', 'balanced_junction']

# Timesteps for training (reduced for quicker results)
timesteps = 50000

# Results tracking - pre-populate with the models you've already trained
results = []
for algorithm in algorithms:
    for reward_type in reward_types:
        model_name = f"{algorithm}_{reward_type}"
        results.append({
            'algorithm': algorithm,
            'reward_type': reward_type,
            'model_name': model_name
        })

# Skip the training part completely
print("Skipping training step since models have already been trained...")

# Evaluate all models
print("\nStep 3: Evaluating all trained models...")
for result in results:
    algorithm = result['algorithm']
    reward_type = result['reward_type']
    model_name = result['model_name']
    
    print(f"\nEvaluating {model_name}...")
    
    # Evaluate the model - using V2_evaluate_models.py instead of evaluate_models.py
    subprocess.run([
        'python3', 'V2_evaluate_models.py',
        '--model', model_name,
        '--algorithm', algorithm,
        '--reward', reward_type,
        '--episodes', '3'  # Reduced for quicker evaluation
    ])
    
    print(f"Completed evaluation of {model_name}")

# Compare all results
print("\nStep 4: Comparing all evaluation results...")
# Using V2_compare_all_results.py instead of compare_all_results.py
subprocess.run(['python3', 'V2_compare_all_results.py'])
print("\nAll experiments completed! Check the comparison results and graphs.")