# custom_ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# A simple Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_outputs),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

class SimplePPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, clip_ratio=0.2, epochs=10):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        # Initialize policy network
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.n
        self.policy = ActorCritic(num_inputs, num_outputs)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def learn(self, total_timesteps):
        # Training loop
        timestep = 0
        episode = 0
        
        while timestep < total_timesteps:
            # Reset environment
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            # Lists to store episode data
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            
            # Collect experience for one episode
            while not done and timestep < total_timesteps:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get action and value
                with torch.no_grad():
                    dist, value = self.policy(state_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                # Take action in environment
                next_state, reward, done, _ = self.env.step(action.item())
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                log_probs.append(log_prob)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                timestep += 1
            
            episode += 1
            print(f"Episode {episode}, Reward: {episode_reward}, Timesteps: {timestep}")
            
            # Calculate advantages and returns
            returns = self._compute_returns(rewards, dones)
            advantages = returns - torch.cat(values).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Optimize policy
            self._update_policy(states, actions, log_probs, returns, advantages)
    
    def _compute_returns(self, rewards, dones):
        returns = []
        R = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def _update_policy(self, states, actions, old_log_probs, returns, advantages):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs).detach()
        
        # Update for n epochs
        for _ in range(self.epochs):
            # Get new log probs and values
            dist, values = self.policy(states)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio and clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            ppo_loss = -torch.min(ratio * advantages, clip_adv).mean()
            
            # Value loss
            value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = ppo_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

# Example usage
def train_custom_ppo(reward_type='simple', total_timesteps=50000):
    from gym.envs.registration import register
    
    try:
        register(
            id="CustomSumo-v0",
            entry_point="sumo_env:SumoEnv",
            max_episode_steps=200,
        )
    except:
        pass
    
    env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=200, reward_type=reward_type)
    ppo = SimplePPO(env)
    ppo.learn(total_timesteps)
    ppo.save(f"custom_ppo_{reward_type}.pt")
    env.close()