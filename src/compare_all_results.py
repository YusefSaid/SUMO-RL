import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# List of result files to analyze
result_files = {
    "RL Model": "tripinfo.xml",
    "60s Fixed Cycle": "baseline_60s_tripinfo.xml",
    "90s Fixed Cycle": "baseline_90s_tripinfo.xml",
    "120s Fixed Cycle": "baseline_120s_tripinfo.xml"
}

# Function to analyze a single tripinfo file
def analyze_tripinfo(file_path, label):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None
    
    print(f"\nAnalyzing {label} from {file_path}")
    
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract trip information
        trips = root.findall("tripinfo")
        
        if not trips:
            print("No trip information found in the file")
            return None
        
        print(f"Found {len(trips)} completed vehicle trips")
        
        # Collect metrics
        durations = []
        waiting_times = []
        time_loss = []
        
        for trip in trips:
            # Extract attributes
            duration = float(trip.get("duration", 0))
            waiting = float(trip.get("waitingTime", 0))
            loss = float(trip.get("timeLoss", 0))
            
            durations.append(duration)
            waiting_times.append(waiting)
            time_loss.append(loss)
        
        # Calculate statistics
        results = {
            "label": label,
            "total_vehicles": len(trips),
            "avg_travel_time": np.mean(durations),
            "avg_waiting_time": np.mean(waiting_times),
            "avg_time_loss": np.mean(time_loss),
            "max_travel_time": np.max(durations),
            "max_waiting_time": np.max(waiting_times),
            "time_loss_percent": 100 * np.mean(time_loss) / np.mean(durations),
            "waiting_distribution": {
                "0-30": sum(1 for w in waiting_times if 0 <= w < 30),
                "30-60": sum(1 for w in waiting_times if 30 <= w < 60),
                "60-120": sum(1 for w in waiting_times if 60 <= w < 120),
                ">120": sum(1 for w in waiting_times if w >= 120)
            }
        }
        
        # Print results
        print(f"Traffic Performance for {label}:")
        print(f"Total vehicles: {results['total_vehicles']}")
        print(f"Average travel time: {results['avg_travel_time']:.2f} seconds")
        print(f"Average waiting time: {results['avg_waiting_time']:.2f} seconds")
        print(f"Average time loss: {results['avg_time_loss']:.2f} seconds")
        print(f"Maximum travel time: {results['max_travel_time']:.2f} seconds")
        print(f"Maximum waiting time: {results['max_waiting_time']:.2f} seconds")
        print(f"Average time loss percentage: {results['time_loss_percent']:.2f}%")
        
        distribution = results['waiting_distribution']
        total = sum(distribution.values())
        
        print("Waiting Time Distribution:")
        for range_label, count in distribution.items():
            if total > 0:
                percentage = 100 * count / total
                print(f"{range_label} seconds: {count} vehicles ({percentage:.2f}%)")
        
        return results
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

# Analyze all files
results = {}
for label, file_path in result_files.items():
    result = analyze_tripinfo(file_path, label)
    if result:
        results[label] = result

# If we have at least two results, compare them
if len(results) >= 2:
    print("\n=== Comparison of All Traffic Light Controllers ===")
    print("Controller\tAvg Wait (s)\tAvg Loss (s)\tLoss %\t% Under 30s")
    
    for label, result in results.items():
        wait = result['avg_waiting_time']
        loss = result['avg_time_loss']
        loss_pct = result['time_loss_percent']
        under_30 = result['waiting_distribution']['0-30']
        total = result['total_vehicles']
        pct_under_30 = 100 * under_30 / total if total > 0 else 0
        
        print(f"{label}\t{wait:.2f}\t{loss:.2f}\t{loss_pct:.2f}%\t{pct_under_30:.2f}%")
    
    # Highlight improvements
    if "RL Model" in results:
        rl = results["RL Model"]
        for label, result in results.items():
            if label != "RL Model":
                wait_diff = result['avg_waiting_time'] - rl['avg_waiting_time']
                loss_diff = result['avg_time_loss'] - rl['avg_time_loss']
                
                if wait_diff > 0:
                    pct_wait_improved = 100 * wait_diff / result['avg_waiting_time']
                    print(f"\nRL model reduced waiting time by {wait_diff:.2f}s ({pct_wait_improved:.2f}%) compared to {label}")
                
                if loss_diff > 0:
                    pct_loss_improved = 100 * loss_diff / result['avg_time_loss']
                    print(f"RL model reduced time loss by {loss_diff:.2f}s ({pct_loss_improved:.2f}%) compared to {label}")

# Plot comparison graph
if len(results) >= 2:
    # Plot waiting time comparison
    plt.figure(figsize=(10, 6))
    labels = list(results.keys())
    waiting_times = [results[label]['avg_waiting_time'] for label in labels]
    
    plt.bar(labels, waiting_times)
    plt.title('Average Waiting Time Comparison')
    plt.ylabel('Average Waiting Time (seconds)')
    plt.savefig('waiting_time_comparison.png')
    print("\nComparison graph saved as 'waiting_time_comparison.png'")