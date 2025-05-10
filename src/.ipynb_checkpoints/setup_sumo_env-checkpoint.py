# setup_sumo_env.py
import os
import sys

# Define SUMO HOME path - adjust this to your SUMO installation
SUMO_HOME = '/home/yrsaid18/Sumo-RL-Project/SUMO-RL/firsttraining/sumo_env'  # Update this path

# Set environment variable
os.environ['SUMO_HOME'] = SUMO_HOME

# Add SUMO tools to Python path
tools_path = os.path.join(SUMO_HOME, 'tools')
if os.path.exists(tools_path):
    sys.path.append(tools_path)
else:
    print(f"❌ SUMO tools path not found at: {tools_path}")
    print("Please check your SUMO installation and update the path.")

# Test if sumolib is available
try:
    import sumolib
    print("✅ sumolib imported successfully!")
    print(f"sumolib path: {sumolib.__file__}")
except ImportError:
    print("❌ Failed to import sumolib. Please check your SUMO installation.")
    # List available paths to help troubleshoot
    print("\nPython path entries:")
    for p in sys.path:
        print(f"  - {p}")