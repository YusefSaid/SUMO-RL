# reward_functions.py
import sys
import os

# Add SUMO tools to Python path
sumo_tools = "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/tools"
if os.path.exists(sumo_tools):
    sys.path.append(sumo_tools)

# Now import SUMO modules
import traci
import sumolib

def simple_waiting_reward(env, obs, action):
    """Current simple reward - negative sum of waiting times"""
    return -float(obs.sum())

def multi_component_reward(env, obs, action, prev_action=None):
    """Multi-component reward with throughput bonuses and switching penalties"""
    # Get edge IDs for tracking vehicles
    incoming_edges = env.incoming_edges
    
    # Initialize reward components
    free_flow_reward = 0
    cleared_halted_reward = 0
    waiting_penalty = -0.05 * float(obs.sum())
    switch_penalty = 0
    
    # Check for phase switching penalty
    if prev_action is not None and action != prev_action:
        switch_penalty = -0.1
    
    # Count vehicles that passed through without stopping
    for edge in incoming_edges:
        # This requires tracking vehicle IDs and their waiting status
        # which would need to be maintained across steps
        # For simplicity, we're using edge statistics
        # A more accurate implementation would track individual vehicles
        current_vehicles = traci.edge.getLastStepVehicleIDs(edge)
        for veh_id in current_vehicles:
            wait_time = traci.vehicle.getWaitingTime(veh_id)
            if wait_time == 0:
                free_flow_reward += 0.5  # Vehicle passing without stopping
            elif wait_time > 0 and traci.vehicle.getSpeed(veh_id) > 0.1:
                cleared_halted_reward += 0.1  # Previously halted vehicle now moving
    
    # Combine all reward components
    total_reward = free_flow_reward + cleared_halted_reward + waiting_penalty + switch_penalty
    
    return total_reward

def difference_reward(env, obs, prev_obs):
    """Reward based on improvement in waiting time"""
    if prev_obs is None:
        return 0.0
    
    # Calculate difference in waiting times
    prev_waiting = float(prev_obs.sum())
    current_waiting = float(obs.sum())
    
    # Reward improvement, penalize worsening
    improvement = prev_waiting - current_waiting
    
    # Scale the reward to be comparable with other reward functions
    scaled_reward = 0.1 * improvement
    
    return scaled_reward

def traffic_flow_reward(env, obs, action):
    """Prioritizes maintaining traffic flow and reducing stops"""
    incoming_edges = env.incoming_edges
    
    # Base penalty for waiting time (smaller weight than standard)
    waiting_penalty = -0.05 * float(obs.sum())
    
    # Track vehicles that are moving vs. stopped
    flow_reward = 0
    stopped_count = 0
    moving_count = 0
    
    for edge in incoming_edges:
        vehicles = traci.edge.getLastStepVehicleIDs(edge)
        for veh_id in vehicles:
            speed = traci.vehicle.getSpeed(veh_id)
            if speed < 0.1:  # Vehicle is stopped
                stopped_count += 1
            else:
                moving_count += 1
                flow_reward += min(speed / 10.0, 1.0)  # Normalize and cap reward
    
    # Congestion penalty (exponential to punish gridlock)
    total_vehicles = stopped_count + moving_count
    congestion_factor = stopped_count / (total_vehicles + 0.001)
    congestion_penalty = -0.5 * (congestion_factor ** 2)
    
    # Combine rewards with emphasis on flow
    total_reward = waiting_penalty + 0.2 * flow_reward + congestion_penalty
    
    return total_reward

def balanced_junction_reward(env, obs, action, prev_action=None):
    """Balances waiting times across all approaches and minimizes overall delay"""
    
    # Get waiting time on each edge
    waiting_times = obs
    
    # Calculate balance metrics
    if len(waiting_times) > 1:
        mean_waiting = waiting_times.mean()
        std_waiting = waiting_times.std() 
        
        # Penalize both high average waiting time and imbalance
        waiting_penalty = -0.1 * mean_waiting
        balance_penalty = -0.2 * std_waiting  # Stronger penalty for imbalance
    else:
        waiting_penalty = -0.1 * float(waiting_times.sum())
        balance_penalty = 0
    
    # Count moving vehicles (throughput)
    throughput_reward = 0
    for edge in env.incoming_edges:
        vehicles = traci.edge.getLastStepVehicleIDs(edge)
        for veh_id in vehicles:
            if traci.vehicle.getSpeed(veh_id) > 0.5:  # Only count if actually moving
                throughput_reward += 0.1
    
    # Small penalty for switching phases too frequently
    switch_penalty = 0
    if prev_action is not None and action != prev_action:
        switch_penalty = -0.1
    
    # Combine rewards
    total_reward = waiting_penalty + balance_penalty + throughput_reward + switch_penalty
    
    return total_reward