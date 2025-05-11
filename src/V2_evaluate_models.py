import os
import sys

##
## 1) make sure we're in the right conda/venv
##
if not hasattr(sys, "real_prefix") and not sys.base_prefix != sys.prefix:
    print("WARNING: you might not be in your sumo_env virtualenv!")

# 2) set SUMO_HOME once and for all
os.environ["SUMO_HOME"] = "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env"

## 3) optional: guard against missing SAC import
try:
    from sb3_contrib import SAC
except ImportError:
    SAC = None

# Register environment if needed
# Import statement moved below the SAC import check
from stable_baselines3 import PPO, A2C

def run_evaluation(model_path, algorithm, reward_type, episodes=5):
    # build the actual filename SB3 expects
    # (SB3 will append ".zip" if you don't give it one)
    fname = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(fname):
        print(f"  ➜  Model file {fname} not found, skipping.")
        return None

    # finally load
    alg = algorithm.lower()
    if alg == 'ppo':
        model = PPO.load(fname, env=env)
    elif alg == 'a2c':
        model = A2C.load(fname, env=env)
    elif alg == 'sac':
        if SAC is None:
            raise RuntimeError("sb3_contrib.SAC not installed in this env!")
        model = SAC.load(fname, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")