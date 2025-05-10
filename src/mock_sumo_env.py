# mock_sumo_env.py
import numpy as np
import gym
from gym import spaces

class MockSumoEnv(gym.Env):
    """A mock environment that simulates a traffic junction without requiring SUMO."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, reward_type='simple', max_steps=200):
        super().__init__()
        
        # Define environment
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.step_count = 0
        
        # Define the incoming edges (lanes) in the junction
        self.incoming_edges = ['edge1', 'edge2', 'edge3', 'edge4']
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=100.0,  # Waiting times can go up to 100 seconds
            shape=(len(self.incoming_edges),),
            dtype=np.float32
        )
        
        # Four actions representing different traffic light phases
        self.action_space = spaces.Discrete(4)
        
        # State variables
        self.current_state = None
        self.prev_state = None
        self.prev_action = None
        
        # Traffic simulator state (simplified)
        self.traffic_state = {
            edge: {
                'waiting_time': 0.0,
                'vehicle_count': 0,
                'vehicle_speeds': []
            }
            for edge in self.incoming_edges
        }
    
    def _get_obs(self):
        """Return the current observation (waiting times on each edge)."""
        return np.array([
            self.traffic_state[edge]['waiting_time'] 
            for edge in self.incoming_edges
        ], dtype=np.float32)
    
    def _update_traffic(self, action):
        """Update traffic state based on action."""
        # Reset speeds and counts
        for edge in self.incoming_edges:
            self.traffic_state[edge]['vehicle_count'] = np.random.randint(0, 10)
            self.traffic_state[edge]['vehicle_speeds'] = [
                np.random.uniform(0, 15) for _ in range(self.traffic_state[edge]['vehicle_count'])
            ]
        
        # Update waiting times based on action
        # In this simplified model, the action prioritizes two of the four edges
        green_edges = []
        if action == 0:
            green_edges = ['edge1', 'edge3']
        elif action == 1:
            green_edges = ['edge2', 'edge4']
        elif action == 2:
            green_edges = ['edge1', 'edge2']
        elif action == 3:
            green_edges = ['edge3', 'edge4']
        
        # Edges with green light should have reduced waiting time
        for edge in self.incoming_edges:
            if edge in green_edges:
                # Green light - reduce waiting time and increase speeds
                self.traffic_state[edge]['waiting_time'] = max(
                    0, self.traffic_state[edge]['waiting_time'] - 5.0
                )
            else:
                # Red light - increase waiting time
                self.traffic_state[edge]['waiting_time'] += np.random.uniform(1.0, 3.0)
    
    def reset(self):
        """Reset the environment to initial state."""
        self.step_count = 0
        
        # Initialize traffic
        for edge in self.incoming_edges:
            self.traffic_state[edge]['waiting_time'] = np.random.uniform(0, 10)
            self.traffic_state[edge]['vehicle_count'] = np.random.randint(0, 5)
            self.traffic_state[edge]['vehicle_speeds'] = [
                np.random.uniform(0, 15) for _ in range(self.traffic_state[edge]['vehicle_count'])
            ]
        
        # Get initial observation
        self.current_state = self._get_obs()
        self.prev_state = None
        self.prev_action = None
        
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment."""
        # Update the traffic state based on action
        self._update_traffic(action)
        
        # Get the new observation
        self.prev_state = self.current_state.copy() if self.current_state is not None else None
        self.current_state = self._get_obs()
        
        # Calculate reward based on reward type
        if self.reward_type == 'simple':
            reward = self._simple_reward()
        elif self.reward_type == 'multi_component':
            reward = self._multi_component_reward(action)
        elif self.reward_type == 'difference':
            reward = self._difference_reward()
        else:
            reward = self._simple_reward()
        
        # Update step count
        self.step_count += 1
        
        # Check if done
        done = (self.step_count >= self.max_steps)
        
        # Store the action for future reference
        self.prev_action = action
        
        return self.current_state, reward, done, {}
    
    def _simple_reward(self):
        """Simple reward - negative sum of waiting times."""
        return -float(self.current_state.sum())
    
    def _multi_component_reward(self, action):
        """Multi-component reward with throughput bonuses and switching penalties."""
        # Waiting time penalty
        waiting_penalty = -0.05 * float(self.current_state.sum())
        
        # Switching penalty
        switch_penalty = 0
        if self.prev_action is not None and action != self.prev_action:
            switch_penalty = -0.1
        
        # Flow reward - count vehicles moving at good speed
        flow_reward = 0
        for edge in self.incoming_edges:
            for speed in self.traffic_state[edge]['vehicle_speeds']:
                if speed > 5.0:  # Vehicle is moving well
                    flow_reward += 0.1
        
        return waiting_penalty + switch_penalty + flow_reward
    
    def _difference_reward(self):
        """Reward based on improvement in waiting time."""
        if self.prev_state is None:
            return 0.0
        
        # Calculate difference in waiting times
        prev_waiting = float(self.prev_state.sum())
        current_waiting = float(self.current_state.sum())
        
        # Reward improvement, penalize worsening
        improvement = prev_waiting - current_waiting
        
        # Scale the reward
        reward = 0.1 * improvement
        
        return reward
    
    def close(self):
        """Clean up the environment."""
        pass