import os
import xml.etree.ElementTree as ET
import numpy as np

# Path to trip info file
tripinfo_file = "tripinfo.xml"

if not os.path.exists(tripinfo_file):
    print(f"Error: {tripinfo_file} not found")
    exit(1)

print(f"Analyzing trip information from {tripinfo_file}")

try:
    # Parse the XML file
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()
    
    # Extract trip information
    trips = root.findall("tripinfo")
    
    if not trips:
        print("No trip information found in the file")
        exit(1)
    
    print(f"Found {len(trips)} completed vehicle trips")
    
    # Collect metrics
    durations = []
    waiting_times = []
    time_loss = []
    
    for trip in trips:
        # Extract attributes (convert to float for calculations)
        duration = float(trip.get("duration", 0))
        waiting = float(trip.get("waitingTime", 0))
        loss = float(trip.get("timeLoss", 0))
        
        durations.append(duration)
        waiting_times.append(waiting)
        time_loss.append(loss)
    
    # Calculate statistics
    print("\nTraffic Performance Metrics:")
    print(f"Total vehicles: {len(trips)}")
    print(f"Average travel time: {np.mean(durations):.2f} seconds")
    print(f"Average waiting time: {np.mean(waiting_times):.2f} seconds")
    print(f"Average time loss: {np.mean(time_loss):.2f} seconds")
    print(f"Maximum travel time: {np.max(durations):.2f} seconds")
    print(f"Maximum waiting time: {np.max(waiting_times):.2f} seconds")
    
    # Calculate additional metrics
    print("\nEfficiency Metrics:")
    avg_time_loss_percent = 100 * np.mean(time_loss) / np.mean(durations)
    print(f"Average time loss percentage: {avg_time_loss_percent:.2f}%")
    
    # Calculate distribution of waiting times
    waiting_bins = [0, 30, 60, 120, float('inf')]
    waiting_counts = [0] * (len(waiting_bins) - 1)
    
    for wait in waiting_times:
        for i in range(len(waiting_bins) - 1):
            if waiting_bins[i] <= wait < waiting_bins[i+1]:
                waiting_counts[i] += 1
                break
    
    print("\nWaiting Time Distribution:")
    for i in range(len(waiting_counts)):
        lower = waiting_bins[i]
        upper = waiting_bins[i+1]
        count = waiting_counts[i]
        percentage = 100 * count / len(waiting_times)
        
        if upper == float('inf'):
            print(f"  > {lower} seconds: {count} vehicles ({percentage:.2f}%)")
        else:
            print(f"  {lower}-{upper} seconds: {count} vehicles ({percentage:.2f}%)")
    
except Exception as e:
    print(f"Error analyzing trip information: {e}")