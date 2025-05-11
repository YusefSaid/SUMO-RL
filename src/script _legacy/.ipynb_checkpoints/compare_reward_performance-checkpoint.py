# compare_reward_performance.py

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict

# Path to SUMO tools directory
sumo_tools = "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/tools"
if os.path.exists(sumo_tools):
    sys.path.append(sumo_tools)

try:
    import traci
    import sumolib
except ImportError:
    print("Cannot import SUMO modules. Using simulated mode only.")
    traci = None
    sumolib = None

# Reward functions
def simple_reward(waiting_times):
    """Original simple reward - negative sum of waiting times"""
    return -np.sum(waiting_times)

def multi_component_reward(waiting_times, prev_waiting_times=None, switched_phase=False):
    """Multi-component reward with throughput bonuses and switching penalties"""
    # Waiting time penalty
    waiting_penalty = -0.05 * np.sum(waiting_times)
    
    # Phase switching penalty
    switch_penalty = -0.1 if switched_phase else 0
    
    # Throughput rewards
    free_flow_reward = 0
    cleared_halted_reward = 0
    
    if prev_waiting_times is not None:
        # Count vehicles that might have cleared
        for i in range(len(waiting_times)):
            if waiting_times[i] < prev_waiting_times[i]:
                # Vehicle likely cleared
                free_flow_reward += 0.5
                if prev_waiting_times[i] > 0:
                    # Previously waiting vehicles cleared
                    cleared_halted_reward += 0.1
    
    return waiting_penalty + switch_penalty + free_flow_reward + cleared_halted_reward

def difference_reward(waiting_times, prev_waiting_times):
    """Reward based on improvement in waiting time"""
    if prev_waiting_times is None:
        return 0
    
    prev_sum = np.sum(prev_waiting_times)
    current_sum = np.sum(waiting_times)
    
    return 0.1 * (prev_sum - current_sum)

# Basic policy for testing - simple fixed phase durations or decision rules
class SimplePolicy:
    def __init__(self, policy_type="fixed_cycle", reward_function=simple_reward):
        self.policy_type = policy_type
        self.reward_function = reward_function
        self.current_phase = 0
        self.phase_duration = 20
        self.time_in_phase = 0
        self.total_reward = 0
        self.prev_waiting_times = None
        self.phase_switches = 0
        
    def select_action(self, state, timestep):
        # Fixed cycle policy
        if self.policy_type == "fixed_cycle":
            if self.time_in_phase >= self.phase_duration:
                self.current_phase = (self.current_phase + 1) % 4
                self.time_in_phase = 0
                self.phase_switches += 1
                switched = True
            else:
                switched = False
                
            self.time_in_phase += 1
            return self.current_phase, switched
        
        # Adaptive policy (uses basic heuristic based on waiting times)
        elif self.policy_type == "adaptive":
            # Simple heuristic: switch if waiting time on current phase is low
            # and waiting time on other phases is high
            phase_waiting_times = self.group_waiting_times_by_phase(state)
            current_phase_wait = phase_waiting_times.get(self.current_phase, 0)
            
            # If we've been in this phase for minimum time and there's more waiting
            # on other phases, consider switching
            if self.time_in_phase >= 5:  # minimum phase duration
                other_phases_wait = sum(v for k, v in phase_waiting_times.items() if k != self.current_phase)
                
                if current_phase_wait < 5 and other_phases_wait > 10:
                    # Time to switch to next phase with waiting vehicles
                    for i in range(1, 4):
                        candidate_phase = (self.current_phase + i) % 4
                        if phase_waiting_times.get(candidate_phase, 0) > 0:
                            self.current_phase = candidate_phase
                            self.time_in_phase = 0
                            self.phase_switches += 1
                            switched = True
                            break
                    else:
                        switched = False
                else:
                    switched = False
            else:
                switched = False
                
            self.time_in_phase += 1
            return self.current_phase, switched
    
    def group_waiting_times_by_phase(self, waiting_times):
        """Group waiting times by traffic light phase (simplified)"""
        # This is a placeholder - in reality, you'd need to map edges to phases
        # Here we just divide the waiting times array into 4 groups
        phase_waiting_times = defaultdict(float)
        chunk_size = len(waiting_times) // 4
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 3 else len(waiting_times)
            phase_waiting_times[i] = sum(waiting_times[start_idx:end_idx])
            
        return phase_waiting_times
    
    def update(self, state, reward, next_state, done):
        """Update policy state and track reward"""
        self.total_reward += reward
        self.prev_waiting_times = state
        
    def get_metrics(self):
        """Return policy performance metrics"""
        return {
            "total_reward": self.total_reward,
            "phase_switches": self.phase_switches
        }

def run_simulation_with_reward(reward_function, sumo_cmd=None, sim_steps=500, policy_type="fixed_cycle"):
    """Run a SUMO simulation with a specific reward function"""
    if traci is None or sumo_cmd is None:
        # Run simulated mode
        return run_simulated_test(reward_function, sim_steps, policy_type)
    
    # Start SUMO
    traci.start(sumo_cmd)
    
    # Initialize policy
    policy = SimplePolicy(policy_type, reward_function)
    
    # Metrics to track
    timestep_metrics = {
        "rewards": [],
        "waiting_times": [],
        "cumulative_reward": 0,
        "vehicles_completed": 0,
        "vehicles_waiting": []
    }
    
    # Run simulation
    for step in range(sim_steps):
        # Get current state (waiting times on edges)
        edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        waiting_times = np.array([traci.edge.getWaitingTime(edge) for edge in edges])
        
        # Select action based on policy
        action, switched = policy.select_action(waiting_times, step)
        
        # Apply traffic light phase
        tls_id = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tls_id, action)
        
        # Calculate reward before next step
        if policy.prev_waiting_times is not None:
            if reward_function.__name__ == "simple_reward":
                reward = simple_reward(waiting_times)
            elif reward_function.__name__ == "multi_component_reward":
                reward = multi_component_reward(waiting_times, policy.prev_waiting_times, switched)
            elif reward_function.__name__ == "difference_reward":
                reward = difference_reward(waiting_times, policy.prev_waiting_times)
        else:
            reward = 0
        
        # Execute simulation step
        traci.simulationStep()
        
        # Update policy
        policy.update(waiting_times, reward, None, False)
        
        # Track metrics
        timestep_metrics["rewards"].append(reward)
        timestep_metrics["waiting_times"].append(np.mean(waiting_times))
        timestep_metrics["cumulative_reward"] = policy.total_reward
        timestep_metrics["vehicles_waiting"].append(sum(1 for w in waiting_times if w > 0))
        
        # Track completed vehicles (those that arrived at destination)
        arrived = traci.simulation.getArrivedNumber()
        timestep_metrics["vehicles_completed"] += arrived
    
    # Close SUMO
    traci.close()
    
    # Calculate summary metrics
    summary_metrics = {
        "avg_waiting_time": np.mean(timestep_metrics["waiting_times"]),
        "max_waiting_time": np.max(timestep_metrics["waiting_times"]),
        "total_reward": timestep_metrics["cumulative_reward"],
        "vehicles_completed": timestep_metrics["vehicles_completed"],
        "avg_vehicles_waiting": np.mean(timestep_metrics["vehicles_waiting"])
    }
    
    # Calculate efficiency metric (ratio of waiting time to total time)
    # This is simplified - you may want to use your original efficiency calculation
    total_time = sim_steps  # total simulation time
    total_waiting = np.sum(timestep_metrics["waiting_times"])
    if total_time > 0:
        summary_metrics["efficiency"] = 100 * (1 - (total_waiting / (total_time * len(edges))))
    else:
        summary_metrics["efficiency"] = 0
    
    return timestep_metrics, summary_metrics

def run_simulated_test(reward_function, sim_steps=500, policy_type="fixed_cycle"):
    """Run a simulated test without SUMO"""
    # Initialize policy
    policy = SimplePolicy(policy_type, reward_function)
    
    # Metrics to track
    timestep_metrics = {
        "rewards": [],
        "waiting_times": [],
        "cumulative_reward": 0,
        "vehicles_completed": 0,
        "vehicles_waiting": []
    }
    
    # Generate synthetic waiting time data
    num_edges = 10
    waiting_times_history = []
    
    for step in range(sim_steps):
        # Generate synthetic waiting times with realistic patterns
        if step == 0:
            # Initial waiting times
            waiting_times = np.random.exponential(scale=3.0, size=num_edges)
        else:
            # Update waiting times based on previous step and action
            # Simulate traffic signal effect
            phase, _ = policy.select_action(waiting_times, step)
            
            # Reduce waiting time for edges that have green light
            green_edges = []
            if phase == 0:
                green_edges = [0, 1]
            elif phase == 1:
                green_edges = [2, 3]
            elif phase == 2:
                green_edges = [4, 5]
            elif phase == 3:
                green_edges = [6, 7]
            
            # Update waiting times
            new_waiting_times = waiting_times.copy()
            
            # Reduce waiting times for green edges
            for edge in green_edges:
                if edge < len(new_waiting_times):
                    # Vehicles clear the intersection
                    new_waiting_times[edge] *= 0.5
            
            # Increase waiting times for red edges
            red_edges = [i for i in range(num_edges) if i not in green_edges]
            for edge in red_edges:
                if edge < len(new_waiting_times):
                    # New vehicles arrive and wait
                    new_waiting_times[edge] += np.random.exponential(scale=1.0)
            
            waiting_times = new_waiting_times
        
        waiting_times_history.append(waiting_times.copy())
        
        # Calculate reward
        if step > 0:
            if reward_function.__name__ == "simple_reward":
                reward = simple_reward(waiting_times)
            elif reward_function.__name__ == "multi_component_reward":
                reward = multi_component_reward(waiting_times, waiting_times_history[-2], False)
            elif reward_function.__name__ == "difference_reward":
                reward = difference_reward(waiting_times, waiting_times_history[-2])
        else:
            reward = 0
        
        # Update policy
        policy.update(waiting_times, reward, None, False)
        
        # Track metrics
        timestep_metrics["rewards"].append(reward)
        timestep_metrics["waiting_times"].append(np.mean(waiting_times))
        timestep_metrics["cumulative_reward"] = policy.total_reward
        timestep_metrics["vehicles_waiting"].append(sum(1 for w in waiting_times if w > 0))
        
        # Simulate completed vehicles
        completed = sum(1 for w in waiting_times if w < 0.5)
        timestep_metrics["vehicles_completed"] += completed
    
    # Calculate summary metrics
    summary_metrics = {
        "avg_waiting_time": np.mean(timestep_metrics["waiting_times"]),
        "max_waiting_time": np.max(timestep_metrics["waiting_times"]),
        "total_reward": timestep_metrics["cumulative_reward"],
        "vehicles_completed": timestep_metrics["vehicles_completed"],
        "avg_vehicles_waiting": np.mean(timestep_metrics["vehicles_waiting"])
    }
    
    # Calculate efficiency metric
    total_time = sim_steps
    total_waiting = np.sum(timestep_metrics["waiting_times"])
    if total_time > 0:
        summary_metrics["efficiency"] = 100 * (1 - (total_waiting / (total_time * num_edges)))
    else:
        summary_metrics["efficiency"] = 0
    
    return timestep_metrics, summary_metrics

def compare_reward_functions(sim_steps=500, policy_type="fixed_cycle", sumo_cmd=None):
    """Compare performance metrics across different reward functions"""
    reward_functions = [
        simple_reward,
        multi_component_reward,
        difference_reward
    ]
    
    results = {}
    
    for reward_func in reward_functions:
        print(f"\nTesting {reward_func.__name__}...")
        timestep_metrics, summary_metrics = run_simulation_with_reward(
            reward_func, sumo_cmd, sim_steps, policy_type)
        
        results[reward_func.__name__] = {
            "timestep_metrics": timestep_metrics,
            "summary_metrics": summary_metrics
        }
        
        print(f"Average Waiting Time: {summary_metrics['avg_waiting_time']:.2f} seconds")
        print(f"Efficiency: {summary_metrics['efficiency']:.2f}%")
        print(f"Total Reward: {summary_metrics['total_reward']:.2f}")
    
    return results

def visualize_comparison(results):
    """Create visualizations comparing the performance of different reward functions"""
    reward_functions = list(results.keys())
    
    # Extract key metrics
    avg_waiting_times = [results[r]["summary_metrics"]["avg_waiting_time"] for r in reward_functions]
    efficiencies = [results[r]["summary_metrics"]["efficiency"] for r in reward_functions]
    total_rewards = [results[r]["summary_metrics"]["total_reward"] for r in reward_functions]
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot waiting times
    plt.subplot(2, 2, 1)
    plt.bar(reward_functions, avg_waiting_times)
    plt.title('Average Waiting Time by Reward Function')
    plt.ylabel('Average Waiting Time (s)')
    plt.ylim(bottom=0)
    
    # Plot efficiencies
    plt.subplot(2, 2, 2)
    plt.bar(reward_functions, efficiencies)
    plt.title('Traffic Control Efficiency by Reward Function')
    plt.ylabel('Efficiency (%)')
    plt.ylim(0, 100)
    
    # Plot total rewards
    plt.subplot(2, 2, 3)
    plt.bar(reward_functions, total_rewards)
    plt.title('Total Reward by Reward Function')
    plt.ylabel('Total Reward')
    
    # Plot waiting time over time
    plt.subplot(2, 2, 4)
    for r in reward_functions:
        timesteps = range(len(results[r]["timestep_metrics"]["waiting_times"]))
        plt.plot(timesteps, results[r]["timestep_metrics"]["waiting_times"], label=r)
    plt.title('Waiting Time Over Simulation')
    plt.xlabel('Timestep')
    plt.ylabel('Average Waiting Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('reward_performance_comparison.png')
    print("Visualization saved as 'reward_performance_comparison.png'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare reward function performance')
    parser.add_argument('--steps', type=int, default=500,
                       help='Number of simulation steps')
    parser.add_argument('--policy', type=str, default="fixed_cycle",
                       choices=["fixed_cycle", "adaptive"],
                       help='Policy type to use')
    parser.add_argument('--sumo', action='store_true',
                       help='Run with SUMO (if available)')
    
    args = parser.parse_args()
    
    # Set up SUMO command if requested
    sumo_cmd = None
    if args.sumo and traci is not None:
        sumo_home = os.path.dirname(os.path.dirname(sumo_tools))
        sumo_binary = os.path.join(sumo_home, "bin", "sumo")
        sumo_cmd = [
            sumo_binary,
            "-c", "sim.sumocfg",
            "--no-step-log", "true"
        ]
    
    # Compare reward functions
    results = compare_reward_functions(args.steps, args.policy, sumo_cmd)
    
    # Visualize results
    visualize_comparison(results)