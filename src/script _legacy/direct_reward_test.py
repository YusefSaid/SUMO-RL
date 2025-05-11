# direct_reward_test.py
import os
import sys
import subprocess
import numpy as np

# Add SUMO tools to path
sumo_dirs = [
    "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env",
    "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_compat_env"
]

for sumo_dir in sumo_dirs:
    tools_dir = os.path.join(sumo_dir, "tools")
    if os.path.exists(tools_dir) and tools_dir not in sys.path:
        sys.path.append(tools_dir)
        print(f"Added to Python path: {tools_dir}")

try:
    import traci
    import sumolib
except ImportError as e:
    print(f"Error importing SUMO modules: {e}")
    sys.exit(1)

# Define reward functions directly
def simple_reward(state):
    """Original simple reward - negative sum of waiting times"""
    return -float(sum(state))

def multi_component_reward(state, prev_state=None, switched_phase=False):
    """Multi-component reward function"""
    # Get the total waiting time penalty
    waiting_penalty = -0.05 * float(sum(state))
    
    # Phase switching penalty
    switch_penalty = -0.1 if switched_phase else 0
    
    # Simulating throughput rewards
    # (We'd need vehicle IDs to track this properly, but we'll simulate)
    num_edges = len(state)
    free_flow_reward = 0
    cleared_halted_reward = 0
    
    if prev_state is not None:
        for i in range(num_edges):
            if state[i] < prev_state[i]:
                # Vehicle likely cleared
                free_flow_reward += 0.1
                if prev_state[i] > 0:
                    # Previously had waiting vehicles
                    cleared_halted_reward += 0.1
    
    # Combine components
    return waiting_penalty + switch_penalty + free_flow_reward + cleared_halted_reward

def difference_reward(state, prev_state):
    """Reward based on improvement in waiting time"""
    if prev_state is None:
        return 0
        
    prev_waiting = float(sum(prev_state))
    current_waiting = float(sum(state))
    
    # Reward improvement
    return 0.1 * (prev_waiting - current_waiting)

# Function to run a simple simulation with fixed actions
def run_simulation(reward_function, sim_steps=200):
    # Start SUMO
    sumo_cmd = [os.path.join(sumo_dir, "bin", "sumo"), 
                "-c", "sim.sumocfg",
                "--no-step-log", "true"]
    
    traci.start(sumo_cmd)
    
    # Track metrics
    total_reward = 0
    waiting_times = []
    prev_state = None
    prev_action = None
    
    # Run simulation
    for step in range(sim_steps):
        # Simple fixed action pattern
        action = step % 4  # Assuming 4 traffic light phases
        
        # Check if phase switched
        phase_switched = prev_action is not None and action != prev_action
        
        # Apply traffic light phase
        tls_id = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tls_id, action)
        
        # Execute simulation step
        traci.simulationStep()
        
        # Get current state (waiting times on edges)
        edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        state = [traci.edge.getWaitingTime(edge) for edge in edges]
        
        # Calculate reward
        if reward_function.__name__ == "simple_reward":
            reward = simple_reward(state)
        elif reward_function.__name__ == "multi_component_reward":
            reward = multi_component_reward(state, prev_state, phase_switched)
        elif reward_function.__name__ == "difference_reward":
            reward = difference_reward(state, prev_state) if prev_state else 0
        
        total_reward += reward
        waiting_times.append(sum(state))
        
        # Update previous state and action
        prev_state = state
        prev_action = action
    
    # Close SUMO
    traci.close()
    
    # Calculate metrics
    avg_waiting_time = np.mean(waiting_times)
    max_waiting_time = np.max(waiting_times)
    
    return {
        "total_reward": total_reward,
        "avg_waiting_time": avg_waiting_time,
        "max_waiting_time": max_waiting_time
    }

# Test all reward functions
def test_all_rewards():
    reward_functions = [
        simple_reward,
        multi_component_reward,
        difference_reward
    ]
    
    results = {}
    
    for reward_func in reward_functions:
        print(f"\nTesting {reward_func.__name__}...")
        result = run_simulation(reward_func)
        results[reward_func.__name__] = result
        
        print(f"Total Reward: {result['total_reward']:.2f}")
        print(f"Average Waiting Time: {result['avg_waiting_time']:.2f}")
        print(f"Maximum Waiting Time: {result['max_waiting_time']:.2f}")
    
    # Compare results
    print("\n--- Comparison ---")
    for name, result in results.items():
        print(f"{name}: Reward={result['total_reward']:.2f}, Avg Wait={result['avg_waiting_time']:.2f}")

if __name__ == "__main__":
    test_all_rewards()