import os
import subprocess

# Print current directory
print(f"Current directory: {os.getcwd()}")

# Check SUMO configuration file
sumo_cfg = "sim.sumocfg"
if not os.path.exists(sumo_cfg):
    print(f"Error: {sumo_cfg} not found in current directory")
    print("Files in directory:")
    for file in os.listdir('.'):
        print(f"  {file}")
    exit(1)

# Set SUMO_HOME if not already set
if "SUMO_HOME" not in os.environ:
    sumo_home = "/home/yrsaid18/venvs/sumo-rl/lib/python3.10/site-packages/sumo"
    os.environ["SUMO_HOME"] = sumo_home
    print(f"Set SUMO_HOME to {sumo_home}")
else:
    print(f"Using SUMO_HOME: {os.environ['SUMO_HOME']}")

# Define path to sumo
sumo_path = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
if not os.path.exists(sumo_path):
    print(f"Warning: {sumo_path} not found, trying 'sumo' from PATH")
    sumo_path = "sumo"

# Create command
cmd = [sumo_path, "-c", sumo_cfg, "--tripinfo-output", "tripinfo.xml"]
print(f"Running command: {' '.join(cmd)}")

# Run SUMO simulation
try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Simulation completed successfully")
    
    # Print stdout and stderr if any
    if result.stdout:
        print("\nOutput:")
        print(result.stdout)
    
    if result.stderr:
        print("\nErrors/Warnings:")
        print(result.stderr)
    
    # Check for tripinfo.xml
    if os.path.exists("tripinfo.xml"):
        print("\nTrip information saved to tripinfo.xml")
        print("\nFirst few lines:")
        with open("tripinfo.xml", "r") as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(line.strip())
                else:
                    print("...")
                    break
    
except subprocess.CalledProcessError as e:
    print(f"Error running SUMO: {e}")
    if e.stdout:
        print("Output:", e.stdout)
    if e.stderr:
        print("Error:", e.stderr)
except Exception as e:
    print(f"Unexpected error: {e}")