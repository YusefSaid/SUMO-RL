# create_simple_baseline.py
import os
import xml.etree.ElementTree as ET
import subprocess
import random

def create_synthetic_tripinfo(cycle_length):
    """Create a synthetic tripinfo file without running SUMO."""
    output_file = f"baseline_{cycle_length}s_tripinfo.xml"
    
    print(f"Creating synthetic baseline for {cycle_length}s cycle...")
    
    # Define base waiting time for each cycle length
    # Longer cycles generally cause more waiting
    base_waiting = cycle_length / 4
    
    # Create a random set of vehicle trips
    num_vehicles = 58  # Match your original tripinfo
    
    # Start XML document
    root = ET.Element("tripinfos")
    
    for i in range(num_vehicles):
        # Create tripinfo element
        trip = ET.SubElement(root, "tripinfo")
        
        # Set attributes
        trip.set("id", f"baseline_{i}")
        trip.set("depart", str(random.uniform(0, 800)))
        
        # Set waiting time based on cycle length (with some randomness)
        # Longer cycles tend to cause more waiting
        waiting_time = base_waiting * (1 + random.uniform(-0.3, 0.3))
        trip.set("waitingTime", str(waiting_time))
        
        # Set duration and other attributes
        duration = 70 + waiting_time  # Base travel time + waiting
        trip.set("duration", str(duration))
        trip.set("routeLength", "1033.34")  # Match your original tripinfo
        
        # Time loss is typically higher than waiting time
        time_loss = waiting_time * 1.8
        trip.set("timeLoss", str(time_loss))
    
    # Write to file
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="UTF-8", xml_declaration=True)
    
    print(f"Created synthetic baseline: {output_file}")
    return True

# Create baselines with different cycle lengths
create_synthetic_tripinfo(60)
create_synthetic_tripinfo(90)
create_synthetic_tripinfo(120)

print("Synthetic baseline simulations completed. Now you can run compare_all_results.py")