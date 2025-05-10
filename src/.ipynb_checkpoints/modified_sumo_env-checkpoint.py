# modified_sumo_env.py
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

# Try to find SUMO
potential_sumo_paths = [
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env',
    '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env/tools',
    '/home/yrsaid18/.local/lib/python3.10/site-packages',
    '/usr/share/sumo'
]

for path in potential_sumo_paths:
    if os.path.exists(path):
        if 'SUMO_HOME' not in os.environ:
            os.environ['SUMO_HOME'] = path
        
        # Add tools directory to path
        tools_path = os.path.join(path, 'tools')
        if os.path.exists(tools_path) and tools_path not in sys.path:
            sys.path.append(tools_path)

# Try importing traci and sumolib
try:
    import traci
    from sumolib import checkBinary, net
except ImportError as e:
    print(f"Error importing SUMO modules: {e}")
    print("Attempting to use relative import path...")
    sys.path.append('/home/yrsaid18/.local/lib/python3.10/site-packages')
    try:
        import traci
        from sumolib import checkBinary, net
    except ImportError as e:
        print(f"Still cannot import SUMO modules: {e}")
        raise e

# Import gym but be flexible with versions
try:
    import gym
    from gym import spaces
    GYM_VERSION = gym.__version__
    print(f"Using gym version: {GYM_VERSION}")
    NEW_GYM_VERSION = int(GYM_VERSION.split('.')[0]) > 0
except ImportError:
    print("Could not import gym. Please install it with: pip install gym==0.21.0")
    raise

# Import reward functions
from reward_functions import simple_waiting_reward, multi_component_reward, difference_reward

class SumoEnv(gym.Env):
    """Gym wrapper for SUMO traffic simulation."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, sumo_cfg="sim.sumocfg", max_steps=200, reward_type='simple'):
        super().__init__()
        self.sumo_binary = checkBinary("sumo")
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0
        self.reward_type = reward_type
        self.prev_obs = None
        self.prev_action = None
        
        # Parse sim.sumocfg to find the net file
        try:
            root = ET.parse(sumo_cfg).getroot()
            net_file = root.find("input/net-file").attrib["value"]
            
            # Load the network and list all non-internal edges
            net_obj = net.readNet(net_file)
            self.incoming_edges = [
                e.getID() for e in net_obj.getEdges()
                if not e.getID().startswith(":")
            ]
        except Exception as e:
            print(f"Error parsing SUMO configuration: {e}")
            # Fallback to default edges if configuration fails
            self.incoming_edges = ["edge1", "edge2", "edge3", "edge4"]
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=0.0, high=float('inf'),
            shape=(len(self.incoming_edges),),
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(4)
    
    def reset(self):
        """Reset the environment."""
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
        
        # Warm up one step
        traci.simulationStep()
        
        # Get initial observation
        obs = self._get_obs()
        
        # Handle gym version differences in reset method
        if NEW_GYM_VERSION:
            return obs, {}  # New gym API requires info dict
        else:
            return obs      # Old gym API just returns observation
    
    def step(self, action):
        """Execute the action and advance the simulation."""
        # Apply traffic light phase
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
        else:
            reward = simple_waiting_reward(self, obs, action)
        
        done = (self.step_count >= self.max_steps)
        
        # Additional info dict
        info = {}
        
        # Update state for next step
        self.prev_obs = obs.copy()
        self.prev_action = action
        
        # Handle gym version differences in step method
        if NEW_GYM_VERSION:
            return obs, reward, done, False, info  # New gym API includes truncated flag
        else:
            return obs, reward, done, info         # Old gym API format
    
    def _get_obs(self):
        """Get the current observation (waiting times on edges)."""
        waits = []
        for edgeID in self.incoming_edges:
            try:
                wait_time = traci.edge.getWaitingTime(edgeID)
                waits.append(wait_time)
            except traci.TraCIException:
                # Handle edge not found error
                waits.append(0.0)
        
        return np.array(waits, dtype=np.float32)
    
    def close(self):
        """Close the environment and SUMO."""
        if traci.isLoaded():
            traci.close()