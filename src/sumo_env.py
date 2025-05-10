import os
import sys
import gym
import numpy as np
from gym import spaces
import xml.etree.ElementTree as ET

# Try multiple potential SUMO locations
potential_sumo_paths = [
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env',
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/tools',
    '/home/yrsaid18/.local/lib/python3.10/site-packages',
    '/usr/share/sumo',
    '/opt/sumo'
]

# Try to find SUMO_HOME
for path in potential_sumo_paths:
    if os.path.exists(path):
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = path
            print(f"Set SUMO_HOME to {path}")
        
        # Add tools directory to path
        tools_path = os.path.join(path, 'tools')
        if os.path.exists(tools_path) and tools_path not in sys.path:
            sys.path.append(tools_path)
            print(f"Added {tools_path} to Python path")
        
        # Try import after each path addition
        try:
            import traci
            from sumolib import checkBinary, net
            print("Successfully imported SUMO modules")
            break
        except ImportError:
            continue
else:
    # If we get here, none of the paths worked
    print("❌ Could not find sumolib. Please install SUMO and set SUMO_HOME correctly.")
    # Continue anyway to see detailed import error
    try:
        import traci
        from sumolib import checkBinary, net
    except ImportError as e:
        print(f"Detailed import error: {e}")
        raise e

# Import reward functions
from reward_functions import simple_waiting_reward, multi_component_reward, difference_reward, traffic_flow_reward, balanced_junction_reward

class SumoEnv(gym.Env):
    """Gym wrapper for your custom SUMO network."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, sumo_cfg: str, max_steps: int = 200, reward_type='simple'):
        super().__init__()
        self.sumo_binary = checkBinary("sumo")
        self.sumo_cfg    = sumo_cfg
        self.max_steps   = max_steps
        self.step_count  = 0
        self.reward_type = reward_type
        self.prev_obs = None
        self.prev_action = None
        
        # Parse sim.sumocfg to find the net file
        root = ET.parse(sumo_cfg).getroot()
        net_file = root.find("input/net-file").attrib["value"]
        
        # Load the network and list all non-internal edges
        net_obj = net.readNet(net_file)
        self.incoming_edges = [
            e.getID() for e in net_obj.getEdges()
            if not e.getID().startswith(":")
        ]
        
        # Define spaces
        # Waiting time on each edge, unbounded ≥0
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf,
            shape=(len(self.incoming_edges),),
            dtype=np.float32
        )
        # Assume your TLS has 4 phases; adjust if needed
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        if traci.isLoaded():
            traci.close()
        traci.start([
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--no-step-log", "true"
        ])
        self.step_count = 0
        self.prev_obs = None
        self.prev_action = None
        # Warm up one step so vehicles at depart=0 appear
        traci.simulationStep()
        return self._get_obs()
    
    def step(self, action: int):
        # Apply a light phase
        tls_id = traci.trafficlight.getIDList()[0]
        traci.trafficlight.setPhase(tls_id, action)
        
        traci.simulationStep()
        self.step_count += 1
        
        obs = self._get_obs()
        
        # Select reward function based on type
        if self.reward_type == 'simple':
            reward = simple_waiting_reward(self, obs, action)
        elif self.reward_type == 'multi_component':
            reward = multi_component_reward(self, obs, action, self.prev_action)
        elif self.reward_type == 'difference':
            reward = difference_reward(self, obs, self.prev_obs)
        elif self.reward_type == 'traffic_density':
            reward = traffic_density_reward(self, obs, action)
        elif self.reward_type == 'velocity_based':
            reward = velocity_based_reward(self, obs, action)
        elif self.reward_type == 'queue_length':
            reward = queue_length_reward(self, obs, action)
        elif self.reward_type == 'time_loss':
            reward = time_loss_reward(self, obs, action)
        elif self.reward_type == 'adaptive_weighted':
            reward = adaptive_weighted_reward(self, obs, action, self.prev_action)
        elif self.reward_type == 'emergency_vehicle':
            reward = emergency_vehicle_reward(self, obs, action)
        elif self.reward_type == 'delay_throughput':
            reward = delay_throughput_reward(self, obs, action)
        elif self.reward_type == 'emission_reduction':
            reward = emission_reduction_reward(self, obs, action)
        else:
            # Default to simple reward if unknown type
            reward = simple_waiting_reward(self, obs, action)
        
        done = (self.step_count >= self.max_steps)
        info = {}
        
        # Update state for next step
        self.prev_obs = obs.copy()
        self.prev_action = action
        
        return obs, reward, done, info
    
    def _get_obs(self):
        # waitingTime on each edge
        waits = [
            traci.edge.getWaitingTime(edgeID)
            for edgeID in self.incoming_edges
        ]
        return np.array(waits, dtype=np.float32)
    
    def close(self):
        if traci.isLoaded():
            traci.close()