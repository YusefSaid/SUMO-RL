from gym.envs.registration import register
import gym
from sumo_env import SumoEnv

# 1) register your env
register(
    id="CustomSumo-v0",
    entry_point="sumo_env:SumoEnv",
    max_episode_steps=200,
)

# 2) instantiate & run a short loop
env = gym.make("CustomSumo-v0", sumo_cfg="sim.sumocfg", max_steps=100)
obs = env.reset()
print("Initial obs:", obs)

for step in range(20):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Step {step:2d}: action={action} | obs={obs} | reward={reward:.1f} | done={done}")
    if done:
        break

env.close()
