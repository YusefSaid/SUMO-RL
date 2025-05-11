# test_sumo_setup.py
import sys
import os

# Print current Python path
print("Current PYTHONPATH:")
for path in sys.path:
    print(f"  - {path}")

# Try to find SUMO directories
print("\nLooking for SUMO directories...")
sumo_dirs = [
    "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env",
    "/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_compat_env"
]

for sumo_dir in sumo_dirs:
    tools_dir = os.path.join(sumo_dir, "tools")
    if os.path.exists(tools_dir):
        print(f"Found SUMO tools at: {tools_dir}")
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
            print(f"Added to Python path: {tools_dir}")

# Try importing SUMO modules
print("\nTrying to import SUMO modules...")
try:
    import traci
    print("Successfully imported traci")
except ImportError as e:
    print(f"Failed to import traci: {e}")

try:
    import sumolib
    print("Successfully imported sumolib")
except ImportError as e:
    print(f"Failed to import sumolib: {e}")

# Print SUMO version if available
if 'sumolib' in sys.modules:
    print(f"\nSUMO version: {sumolib.version.get_version()}")